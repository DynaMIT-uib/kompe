"""Spherical Elementary Current Systems as a first-class basis."""

from __future__ import annotations

import numpy as np

from kompe.constants import EARTH_RADIUS_M
from kompe.core import ScalarSynthesis
from kompe.grid import Grid
from kompe.math import as_linear_map
from kompe.secs.kernels import magnetic_field_matrices, surface_current_matrices

DEFAULT_IONOSPHERE_RADIUS_M = EARTH_RADIUS_M + 110e3


class SECSBasis(ScalarSynthesis):
    """A basis of spherical elementary current systems.

    Coefficients are amplitudes at the pole locations in ``poles``. One
    pole layout supports curl-free and divergence-free surface-current
    synthesis, their scalar Green kernels, and magnetic-field synthesis.

    SECS Green functions have distributional surface Laplacians. This
    class therefore implements scalar and vector synthesis, but does not
    claim the square coefficient-space Laplacian and closed-surface gauge
    semantics of :class:`~kompe.SurfaceOperators`.

    Parameters
    ----------
    poles : Grid, optional
        Geographic locations of the elementary systems.
    lat, lon : array-like, optional
        Pole coordinates in degrees when ``poles`` is not supplied.
    radius : float, optional
        Radius of the current sheet. Units are arbitrary but must be
        consistent with evaluation radii and singularity limits.
    constant : float, optional
        Green-function normalization. The standard convention is ``1/(4*pi)``.
    """

    surface_component_order = ("theta", "phi")
    magnetic_component_order = ("radial", "theta", "phi")
    helmholtz_component_order = ("curl_free", "divergence_free")

    def __init__(
        self,
        poles=None,
        *,
        lat=None,
        lon=None,
        radius=DEFAULT_IONOSPHERE_RADIUS_M,
        constant=1.0 / (4.0 * np.pi),
    ):
        if poles is not None and (lat is not None or lon is not None):
            raise ValueError("Provide either poles or lat/lon, not both.")
        if poles is None:
            if lat is None or lon is None:
                raise ValueError("SECSBasis requires poles or both lat and lon.")
            poles = Grid(lat=lat, lon=lon)
        if not isinstance(poles, Grid):
            if all(hasattr(poles, name) for name in ("lat", "lon", "size")):
                poles = Grid(lat=poles.lat, lon=poles.lon)
            else:
                raise TypeError(
                    "SECSBasis poles must be a spherical point grid with lat/lon coordinates."
                )

        radius = float(radius)
        constant = float(constant)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("SECSBasis radius must be finite and positive.")
        if not np.isfinite(constant):
            raise ValueError("SECSBasis constant must be finite.")

        self.poles = poles
        self.radius = radius
        self.constant = constant
        self.validate_metadata()

    @property
    def kind(self):
        """Short identifier for SECS representations."""
        return "SECS"

    @property
    def index_names(self):
        """Names of pole-coordinate metadata arrays."""
        return ("latitude", "longitude")

    @property
    def index_length(self):
        """Number of elementary systems."""
        return self.poles.size

    @property
    def index_arrays(self):
        """Pole latitude and longitude arrays in degrees."""
        return self.poles.lat, self.poles.lon

    @property
    def signature(self):
        """Return exact basis and normalization identity."""
        return ("SECS", self.poles.exact_coordinate_signature, self.radius, self.constant)

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

    def evaluate_on_grid(self, grid, derivative=None):
        """Evaluate the curl-free SECS scalar potential Green functions.

        Derivatives are intentionally exposed through the explicitly framed
        surface-current methods instead of overloading angular derivatives
        with radius-scaled current density.
        """
        self._validate_grid(grid)
        if derivative is not None:
            raise NotImplementedError(
                "SECS potential derivatives are represented by surface-current synthesis."
            )
        return surface_current_matrices(
            grid.lat,
            grid.lon,
            self.poles.lat,
            self.poles.lon,
            current_type="potential",
            constant=self.constant,
            RI=self.radius,
        )

    def get_scalar_kernel_matrix(self, grid):
        """Return the legacy divergence-free scalar kernel matrix."""
        self._validate_grid(grid)
        return surface_current_matrices(
            grid.lat,
            grid.lon,
            self.poles.lat,
            self.poles.lon,
            current_type="scalar",
            constant=self.constant,
            RI=self.radius,
        )

    def get_surface_current_matrix(
        self, grid, *, current_type="divergence_free", singularity_limit=0.0
    ):
        """Return current synthesis in canonical ``(theta, phi)`` order."""
        self._validate_grid(grid)
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

    def get_surface_current_operator(
        self, grid, *, current_type="divergence_free", singularity_limit=0.0
    ):
        """Return coefficient-to-current synthesis as a structured operator."""
        matrix = self.get_surface_current_matrix(
            grid, current_type=current_type, singularity_limit=singularity_limit
        )
        return as_linear_map(matrix, input_shape=(self.index_length,), output_shape=(2, grid.size))

    def get_helmholtz_current_synthesis_matrix(self, grid, *, singularity_limit=0.0):
        """Return curl-free/divergence-free current synthesis tensor."""
        curl_free = self.get_surface_current_matrix(
            grid, current_type="curl_free", singularity_limit=singularity_limit
        )
        divergence_free = self.get_surface_current_matrix(
            grid, current_type="divergence_free", singularity_limit=singularity_limit
        )
        return np.stack([curl_free, divergence_free], axis=2)

    def get_helmholtz_current_synthesis_operator(self, grid, *, singularity_limit=0.0):
        """Return two-potential SECS current synthesis operator."""
        matrix = self.get_helmholtz_current_synthesis_matrix(
            grid, singularity_limit=singularity_limit
        )
        return as_linear_map(
            matrix, input_shape=(2, self.index_length), output_shape=(2, grid.size)
        )

    def get_magnetic_field_matrix(
        self,
        grid,
        evaluation_radius,
        *,
        current_type="divergence_free",
        singularity_limit=0.0,
        induction_nullification_radius=None,
    ):
        """Return magnetic synthesis in canonical ``(radial, theta, phi)`` order."""
        self._validate_grid(grid)
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

    def get_magnetic_field_operator(self, grid, evaluation_radius, **kwargs):
        """Return coefficient-to-magnetic-field synthesis operator."""
        matrix = self.get_magnetic_field_matrix(grid, evaluation_radius, **kwargs)
        return as_linear_map(matrix, input_shape=(self.index_length,), output_shape=(3, grid.size))


__all__ = ["DEFAULT_IONOSPHERE_RADIUS_M", "EARTH_RADIUS_M", "SECSBasis"]
