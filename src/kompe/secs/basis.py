"""Spherical Elementary Current Systems as a first-class basis."""

from __future__ import annotations

import numpy as np

from kompe.constants import EARTH_RADIUS_M
from kompe.core import ScalarBasis
from kompe.grid import SphericalGrid
from kompe.math import LinearMap, as_linear_map
from kompe.secs.kernels import magnetic_field_matrices, surface_current_matrices

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
    constant : float, optional
        Green-function normalization. The standard convention is ``1/(4*pi)``.
    current_type : {"curl_free", "divergence_free"}
        Physical current-system mode represented by scalar coefficients.
    """

    def __init__(
        self,
        poles,
        *,
        radius=DEFAULT_IONOSPHERE_RADIUS_M,
        constant=1.0 / (4.0 * np.pi),
        current_type,
    ):
        if not isinstance(poles, SphericalGrid):
            raise TypeError("SECSBasis poles must be a SphericalGrid.")

        radius = float(radius)
        constant = float(constant)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("SECSBasis radius must be finite and positive.")
        if not np.isfinite(constant):
            raise ValueError("SECSBasis constant must be finite.")
        if current_type not in {"curl_free", "divergence_free"}:
            raise ValueError("current_type must be 'curl_free' or 'divergence_free'.")

        self.poles = poles
        self.radius = radius
        self.constant = constant
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
            f"radius={self.radius:g}, constant={self.constant:g})"
        )

    @property
    def signature(self):
        """Return exact basis and normalization identity."""
        return (
            "SECS",
            self.current_type,
            self.poles.exact_coordinate_signature,
            self.radius,
            self.constant,
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

    def scalar_evaluation_matrix(self, grid, derivative=None):
        """Evaluate scalar Green functions for the selected current-system mode.

        Derivatives are intentionally exposed through the explicitly framed
        surface-current methods instead of overloading angular derivatives
        with radius-scaled current density.
        """
        self._validate_grid(grid)
        if derivative is not None:
            raise NotImplementedError(
                "SECS potential derivatives are represented by surface-current synthesis."
            )
        scalar_type = "potential" if self.current_type == "curl_free" else "scalar"
        return surface_current_matrices(
            grid.lat,
            grid.lon,
            self.poles.lat,
            self.poles.lon,
            current_type=scalar_type,
            constant=self.constant,
            RI=self.radius,
        )

    def surface_current_matrix(self, grid, *, current_type=None, singularity_limit=0.0):
        """Return current synthesis in canonical ``(theta, phi)`` order."""
        self._validate_grid(grid)
        current_type = self.current_type if current_type is None else current_type
        if current_type not in {"curl_free", "divergence_free"}:
            raise ValueError("current_type must be 'curl_free' or 'divergence_free'.")
        east, north = surface_current_matrices(
            grid.lat,
            grid.lon,
            self.poles.lat,
            self.poles.lon,
            current_type=current_type,
            constant=self.constant,
            RI=self.radius,
            singularity_limit=singularity_limit,
        )
        return np.stack([-north, east], axis=0)

    def surface_current_operator(
        self,
        grid,
        *,
        current_type=None,
        singularity_limit=0.0,
        chunk_size=None,
    ):
        """Return coefficient-to-current synthesis as a structured operator.

        ``chunk_size`` selects a matrix-free operator that constructs kernels
        for only that many evaluation points at a time. This bounds temporary
        memory while preserving forward, adjoint, and multiple-RHS actions.
        """
        self._validate_grid(grid)
        current_type = self.current_type if current_type is None else current_type
        if current_type not in {"curl_free", "divergence_free"}:
            raise ValueError("current_type must be 'curl_free' or 'divergence_free'.")
        if chunk_size is None:
            matrix = self.surface_current_matrix(
                grid, current_type=current_type, singularity_limit=singularity_limit
            )
            return as_linear_map(
                matrix, input_shape=(self.index_length,), output_shape=(2, grid.size)
            )

        chunk_size = self._validate_chunk_size(chunk_size)
        evaluation_lat = np.asarray(grid.lat).reshape(-1)
        evaluation_lon = np.asarray(grid.lon).reshape(-1)

        def chunk_matrix(start, stop):
            east, north = surface_current_matrices(
                evaluation_lat[start:stop],
                evaluation_lon[start:stop],
                self.poles.lat,
                self.poles.lon,
                current_type=current_type,
                constant=self.constant,
                RI=self.radius,
                singularity_limit=singularity_limit,
            )
            return np.stack([-north, east], axis=0)

        def slices():
            for start in range(0, grid.size, chunk_size):
                yield start, min(start + chunk_size, grid.size)

        def matvec(coefficients):
            coefficients = np.asarray(coefficients).reshape(self.index_length)
            output = np.empty((2, grid.size), dtype=np.result_type(coefficients, float))
            for start, stop in slices():
                output[:, start:stop] = np.einsum(
                    "cnp,p->cn", chunk_matrix(start, stop), coefficients
                )
            return output.reshape(-1)

        def rmatvec(values):
            values = np.asarray(values).reshape(2, grid.size)
            output = np.zeros(self.index_length, dtype=np.result_type(values, float))
            for start, stop in slices():
                output += np.einsum("cnp,cn->p", chunk_matrix(start, stop), values[:, start:stop])
            return output

        def matmat(coefficients):
            coefficients = np.asarray(coefficients).reshape(self.index_length, -1)
            output = np.empty(
                (2, grid.size, coefficients.shape[1]),
                dtype=np.result_type(coefficients, float),
            )
            for start, stop in slices():
                output[:, start:stop] = np.einsum(
                    "cnp,pk->cnk", chunk_matrix(start, stop), coefficients
                )
            return output.reshape(2 * grid.size, -1)

        def rmatmat(values):
            values = np.asarray(values).reshape(2, grid.size, -1)
            output = np.zeros(
                (self.index_length, values.shape[-1]), dtype=np.result_type(values, float)
            )
            for start, stop in slices():
                output += np.einsum(
                    "cnp,cnk->pk", chunk_matrix(start, stop), values[:, start:stop]
                )
            return output

        return LinearMap(
            shape=(2 * grid.size, self.index_length),
            dtype=np.float64,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            input_shape=(self.index_length,),
            output_shape=(2, grid.size),
        )

    def helmholtz_current_synthesis_matrix(self, grid, *, singularity_limit=0.0):
        """Return curl-free/divergence-free current synthesis tensor."""
        curl_free = self.surface_current_matrix(
            grid, current_type="curl_free", singularity_limit=singularity_limit
        )
        divergence_free = self.surface_current_matrix(
            grid, current_type="divergence_free", singularity_limit=singularity_limit
        )
        return np.stack([curl_free, divergence_free], axis=2)

    def helmholtz_current_synthesis_operator(self, grid, *, singularity_limit=0.0):
        """Return two-potential SECS current synthesis operator."""
        matrix = self.helmholtz_current_synthesis_matrix(grid, singularity_limit=singularity_limit)
        return as_linear_map(
            matrix, input_shape=(2, self.index_length), output_shape=(2, grid.size)
        )

    def magnetic_field_matrix(
        self,
        grid,
        evaluation_radius,
        *,
        current_type=None,
        singularity_limit=0.0,
        induction_nullification_radius=None,
    ):
        """Return magnetic synthesis in canonical ``(radial, theta, phi)`` order."""
        self._validate_grid(grid)
        current_type = self.current_type if current_type is None else current_type
        if current_type not in {"curl_free", "divergence_free"}:
            raise ValueError("current_type must be 'curl_free' or 'divergence_free'.")
        east, north, radial = magnetic_field_matrices(
            grid.lat,
            grid.lon,
            evaluation_radius,
            self.poles.lat,
            self.poles.lon,
            current_type=current_type,
            constant=self.constant,
            RI=self.radius,
            singularity_limit=singularity_limit,
            induction_nullification_radius=induction_nullification_radius,
        )
        return np.stack([radial, -north, east], axis=0)

    def magnetic_field_operator(self, grid, evaluation_radius, **kwargs):
        """Return coefficient-to-magnetic-field synthesis operator."""
        matrix = self.magnetic_field_matrix(grid, evaluation_radius, **kwargs)
        return as_linear_map(matrix, input_shape=(self.index_length,), output_shape=(3, grid.size))


__all__ = ["DEFAULT_IONOSPHERE_RADIUS_M", "EARTH_RADIUS_M", "SECSBasis"]
