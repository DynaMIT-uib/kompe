"""Spherical Elementary Current Systems as a first-class basis."""

from __future__ import annotations

import numpy as np

from kompe.basis import ScalarBasis
from kompe.constants import EARTH_RADIUS_M
from kompe.grid import SphericalGrid
from kompe.math import LinearMap, as_linear_map, get_array_module
from kompe.secs.kernels import (
    magnetic_field_matrices,
    scalar_green_matrix,
    surface_current_matrices,
)

DEFAULT_IONOSPHERE_RADIUS_M = EARTH_RADIUS_M + 110e3


class SECSBasis(ScalarBasis):
    """A basis of spherical elementary current systems.

    Coefficients are amplitudes at the pole locations in ``poles``. One
    pole layout supports curl-free and divergence-free surface-current
    synthesis, their scalar Green kernels, and magnetic-field synthesis. The
    required ``current_type`` selects the physical meaning of one coefficient
    vector; two-potential Helmholtz methods remain explicit about both modes.

    SECS Green functions have distributional surface Laplacians. This
    class therefore implements scalar and vector synthesis, but does not
    claim the square coefficient-space Laplacian and closed-surface gauge
    semantics of :class:`~kompe.SurfaceDifferentialBasis`.

    Parameters
    ----------
    poles : SphericalGrid
        Geographic locations of the elementary systems.
    radius : float, optional
        Radius of the current sheet. Units are arbitrary but must be
        consistent with evaluation radii and singularity limits.
    normalization : float, optional
        Green-function normalization. The standard convention is ``1/(4*pi)``.
    current_type : {"curl_free", "divergence_free"}
        Physical current-system mode represented by scalar coefficients.
    """

    def __init__(
        self,
        poles,
        *,
        radius=DEFAULT_IONOSPHERE_RADIUS_M,
        normalization=1.0 / (4.0 * np.pi),
        current_type,
    ):
        if not isinstance(poles, SphericalGrid):
            raise TypeError("SECSBasis poles must be a SphericalGrid.")

        radius = float(radius)
        normalization = float(normalization)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("SECSBasis radius must be finite and positive.")
        if not np.isfinite(normalization):
            raise ValueError("SECSBasis normalization must be finite.")
        if current_type not in {"curl_free", "divergence_free"}:
            raise ValueError("current_type must be 'curl_free' or 'divergence_free'.")

        self.poles = poles
        self.radius = radius
        self.normalization = normalization
        self.current_type = current_type
        self.kind = "SECS"
        self.index_names = ("latitude", "longitude")
        self.index_length = self.poles.size
        self.index_arrays = (self.poles.lat, self.poles.lon)
        self.validate_metadata()

    def __repr__(self):
        """Summarize the elementary-current coefficient space."""
        return (
            f"SECSBasis(current_type={self.current_type!r}, poles={self.poles.size}, "
            f"radius={self.radius:g}, normalization={self.normalization:g})"
        )

    @property
    def signature(self):
        """Return exact basis and normalization identity."""
        return (
            "SECS",
            self.current_type,
            self.poles.signature,
            self.radius,
            self.normalization,
        )

    @property
    def coefficient_space_signature(self):
        """Return coefficient compatibility identity."""
        return self.signature

    @staticmethod
    def _validate_grid(grid):
        if not all(hasattr(grid, name) for name in ("lat", "lon", "size")):
            raise TypeError(
                "SECS evaluation requires a spherical point grid with lat/lon coordinates."
            )
        if np.asarray(grid.lat).size != grid.size or np.asarray(grid.lon).size != grid.size:
            raise ValueError("SECS grid coordinates must contain exactly grid.size points.")
        return grid

    @staticmethod
    def _validate_chunk_size(chunk_size):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        return int(chunk_size)

    def scalar_evaluation_array(self, grid, derivative=None):
        """Evaluate scalar Green functions for the selected current-system mode.

        Derivatives are intentionally exposed through the explicitly framed
        surface-current methods instead of overloading angular derivatives
        with radius-scaled current density.
        """
        self._validate_grid(grid)
        if derivative is not None:
            raise NotImplementedError(
                "SECS scalar derivatives are represented by surface-current synthesis."
            )
        quantity = "potential" if self.current_type == "curl_free" else "current_magnitude"
        return scalar_green_matrix(
            grid.lat,
            grid.lon,
            self.poles.lat,
            self.poles.lon,
            quantity=quantity,
            normalization=self.normalization,
        )

    def surface_current_array(self, grid, *, singularity_limit=0.0):
        """Return this basis's current synthesis in ``(theta, phi)`` order."""
        self._validate_grid(grid)
        east, north = surface_current_matrices(
            grid.lat,
            grid.lon,
            self.poles.lat,
            self.poles.lon,
            current_type=self.current_type,
            normalization=self.normalization,
            source_radius=self.radius,
            singularity_limit=singularity_limit,
        )
        xp = get_array_module(east, north)
        return xp.stack([-north, east], axis=0)

    def surface_current_operator(
        self,
        grid,
        *,
        singularity_limit=0.0,
        chunk_size=None,
    ):
        """Return coefficient-to-current synthesis as a structured operator.

        ``chunk_size`` selects a matrix-free operator that constructs kernels
        for only that many evaluation points at a time. This bounds temporary
        memory while preserving forward, adjoint, and multiple-RHS actions.
        """
        self._validate_grid(grid)
        if chunk_size is None:
            array = self.surface_current_array(grid, singularity_limit=singularity_limit)
            return as_linear_map(
                array, input_shape=(self.index_length,), output_shape=(2, grid.size)
            )

        chunk_size = self._validate_chunk_size(chunk_size)
        evaluation_lat = np.asarray(grid.lat).reshape(-1)
        evaluation_lon = np.asarray(grid.lon).reshape(-1)

        def chunk_matrix(start, stop, xp):
            east, north = surface_current_matrices(
                xp.asarray(evaluation_lat[start:stop]),
                xp.asarray(evaluation_lon[start:stop]),
                self.poles.lat,
                self.poles.lon,
                current_type=self.current_type,
                normalization=self.normalization,
                source_radius=self.radius,
                singularity_limit=singularity_limit,
            )
            return xp.stack([-north, east], axis=0)

        def slices():
            for start in range(0, grid.size, chunk_size):
                yield start, min(start + chunk_size, grid.size)

        def matvec(coefficients):
            xp = get_array_module(coefficients)
            coefficients = xp.asarray(coefficients).reshape(self.index_length)
            output = xp.empty((2, grid.size), dtype=xp.result_type(coefficients, float))
            for start, stop in slices():
                chunk = xp.einsum("cnp,p->cn", chunk_matrix(start, stop, xp), coefficients)
                if xp is np:
                    output[:, start:stop] = chunk
                else:
                    output = output.at[:, start:stop].set(chunk)
            return output.reshape(-1)

        def rmatvec(values):
            xp = get_array_module(values)
            values = xp.asarray(values).reshape(2, grid.size)
            output = xp.zeros(self.index_length, dtype=xp.result_type(values, float))
            for start, stop in slices():
                output = output + xp.einsum(
                    "cnp,cn->p", chunk_matrix(start, stop, xp), values[:, start:stop]
                )
            return output

        def matmat(coefficients):
            xp = get_array_module(coefficients)
            coefficients = xp.asarray(coefficients).reshape(self.index_length, -1)
            output = xp.empty(
                (2, grid.size, coefficients.shape[1]),
                dtype=xp.result_type(coefficients, float),
            )
            for start, stop in slices():
                chunk = xp.einsum("cnp,pk->cnk", chunk_matrix(start, stop, xp), coefficients)
                if xp is np:
                    output[:, start:stop] = chunk
                else:
                    output = output.at[:, start:stop].set(chunk)
            return output.reshape(2 * grid.size, -1)

        def rmatmat(values):
            xp = get_array_module(values)
            values = xp.asarray(values).reshape(2, grid.size, -1)
            output = xp.zeros(
                (self.index_length, values.shape[-1]), dtype=xp.result_type(values, float)
            )
            for start, stop in slices():
                output = output + xp.einsum(
                    "cnp,cnk->pk", chunk_matrix(start, stop, xp), values[:, start:stop]
                )
            return output

        return LinearMap(
            shape=(2 * grid.size, self.index_length),
            dtype=np.float64,
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            input_shape=(self.index_length,),
            output_shape=(2, grid.size),
        )

    def helmholtz_current_synthesis_array(self, grid, *, singularity_limit=0.0):
        """Return both current modes using this basis's pole geometry."""
        self._validate_grid(grid)
        curl_free_east, curl_free_north = surface_current_matrices(
            grid.lat,
            grid.lon,
            self.poles.lat,
            self.poles.lon,
            current_type="curl_free",
            normalization=self.normalization,
            source_radius=self.radius,
            singularity_limit=singularity_limit,
        )
        divergence_free_east, divergence_free_north = surface_current_matrices(
            grid.lat,
            grid.lon,
            self.poles.lat,
            self.poles.lon,
            current_type="divergence_free",
            normalization=self.normalization,
            source_radius=self.radius,
            singularity_limit=singularity_limit,
        )
        xp = get_array_module(curl_free_east, divergence_free_east)
        curl_free = xp.stack([-curl_free_north, curl_free_east], axis=0)
        divergence_free = xp.stack([-divergence_free_north, divergence_free_east], axis=0)
        return xp.stack([curl_free, divergence_free], axis=2)

    def helmholtz_current_synthesis_operator(self, grid, *, singularity_limit=0.0):
        """Return two-potential SECS current synthesis operator."""
        array = self.helmholtz_current_synthesis_array(grid, singularity_limit=singularity_limit)
        return as_linear_map(
            array, input_shape=(2, self.index_length), output_shape=(2, grid.size)
        )

    def magnetic_field_array(
        self,
        grid,
        evaluation_radius,
        *,
        singularity_limit=0.0,
        induction_nullification_radius=None,
    ):
        """Return this basis's magnetic synthesis in ``(radial, theta, phi)`` order."""
        self._validate_grid(grid)
        east, north, radial = magnetic_field_matrices(
            grid.lat,
            grid.lon,
            evaluation_radius,
            self.poles.lat,
            self.poles.lon,
            current_type=self.current_type,
            normalization=self.normalization,
            source_radius=self.radius,
            singularity_limit=singularity_limit,
            induction_nullification_radius=induction_nullification_radius,
        )
        xp = get_array_module(east, north, radial)
        return xp.stack([radial, -north, east], axis=0)

    def magnetic_field_operator(self, grid, evaluation_radius, **kwargs):
        """Return coefficient-to-magnetic-field synthesis operator."""
        array = self.magnetic_field_array(grid, evaluation_radius, **kwargs)
        return as_linear_map(array, input_shape=(self.index_length,), output_shape=(3, grid.size))


__all__ = ["DEFAULT_IONOSPHERE_RADIUS_M", "EARTH_RADIUS_M", "SECSBasis"]
