"""Cubed sphere basis module.

This module contains the global cubed-sphere surface basis.
basis.
"""

from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

from kompe.core import SurfaceOperators
from kompe.cubed_sphere import cs_coordinates, cs_vectors
from kompe.cubed_sphere.cs_differencing import CSFiniteDifferences
from kompe.cubed_sphere.cs_grid import CSGridRemapper, GlobalCSMesh
from kompe.math import as_linear_map, identity_linear_map
from kompe.math.backend import get_array_module, to_numpy, use_jax
from kompe.math.least_squares_solver import sparse_constrained_least_squares_map


class GlobalCSBasis(SurfaceOperators):
    """Class for representing cubed sphere bases.

    This module provides an implementation of the cubed sphere grid
    system following methods from Yin et al. (2017). The cubed sphere
    grid divides a sphere into six faces of a circumscribed cube,
    providing nearly uniform grid resolution and avoiding pole
    singularities. Each face uses a local (xi, eta) coordinate system
    mapped to global spherical coordinates (theta, phi). It includes
    tools for coordinate transformations, scalar and vector field
    interpolation and manipulation, numerical differentiation, and
    visualization utilities.

    Native CS coefficients are stored at cell centers. Cell areas are
    computed from the surrounding mapped cell corners, while
    differential operators act on cell-centered values and return
    cell-centered derivatives.

    Attributes
    ----------
    N : int
        Number of grid cells per cube edge (only set if N provided in
        constructor).
    arr_xi : ndarray
        Xi coordinates of native cell centers, in radians.
    arr_eta : ndarray
        Eta coordinates of native cell centers, in radians.
    arr_theta : ndarray
        Colatitude coordinates of native cell centers, in degrees.
    arr_phi : ndarray
        Longitude coordinates of native cell centers, in degrees.
    arr_block : ndarray
        Block indices (0-5) of native cell centers.
    g : ndarray
        Metric tensor
    sqrt_detg : ndarray
        Square root of determinant of the metric tensor.
    unit_area : ndarray
        Spherical quadrilateral area of each unit-sphere grid cell,
        computed from mapped cell corners.

    Notes
    -----
    The cubed sphere grid is organized into six faces as shown below,
    which defines the block structure of the grid::

              _______
              |     |
              |  V  |
        ______|_____|____________
        |     |     |     |     |
        | IV  |  I  | II  | III |
        |_____|_____|_____|_____|
              |     |
              | VI  |
              |_____|

    Block indices:

    - 0 = I: Equator
    - 1 = II: Equator
    - 2 = III: Equator
    - 3 = IV: Equator
    - 4 = V: North Pole
    - 5 = VI: South Pole

    References
    ----------
    [1] Liang Yin, Chao Yang, Shi-Zhuang Ma, Ji-Zu Huang, Ying Cai
        (2017) Parallel numerical simulation of the thermal convection
        in the Earth's outer core on the cubed-sphere. Geophysical
        Journal International, 209(3), 1934–1954.
        DOI: 10.1093/gji/ggx125
    """

    _surface_cache_size = 16

    def __init__(self, N=None):
        """Initialize the cubed sphere basis.

        If N is provided, initializes arrays for a grid with N×N cells
        on each cube face. The native coefficients live at the 6×N×N
        cell centers.

        Parameters
        ----------
        N : int, optional
            Number of grid cells per cube edge. Must be even if
            provided.

        Raises
        ------
        TypeError
            If N is provided but is not an integer.
        ValueError
            If N is provided but is not an even number.
        """
        self._kind = "CS"
        self._index_names = None
        self._index_length = None
        self._index_arrays = None
        self._derivative_bundle = None
        self._laplacian_cache = {}
        self._laplacian_sparse_cache = {}
        self._remap_operator_cache = {}
        self._grid_remapper = CSGridRemapper(self, self._remap_operator_cache)
        self._finite_differences = CSFiniteDifferences(self)
        self._surface_matrix_cache = OrderedDict()
        self._surface_operator_cache = OrderedDict()

        if N is not None:
            if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
                raise TypeError("N must be an integer")
            if N <= 0:
                raise ValueError("Cubed sphere grid dimension must be positive")
            if N % 2 != 0:
                raise ValueError("Cubed sphere grid dimension must be even")

            self.N = int(N)
            self.mesh = GlobalCSMesh(self.N)
            self.grid_geometry = self.mesh
            self.arr_xi = self.mesh.arr_xi
            self.arr_eta = self.mesh.arr_eta
            self.arr_block = self.mesh.arr_block
            self.arr_theta = self.mesh.arr_theta
            self.arr_phi = self.mesh.arr_phi
            self.g = self.mesh.metric_tensor
            self.sqrt_detg = self.mesh.sqrt_detg
            self.unit_area = self.mesh.unit_area

            self.index_names = ("theta", "phi")
            self.index_length = self.mesh.index_length
            self.index_arrays = (self.arr_theta, self.arr_phi)

            self.validate_metadata()

    @property
    def kind(self):
        """Short identifier for the cubed-sphere basis."""
        return self._kind

    @property
    def index_names(self):
        """Names of indices used in the basis."""
        return self._index_names

    @index_names.setter
    def index_names(self, value):
        self._index_names = tuple(value)

    @property
    def index_length(self):
        """Total number of native CS coefficients."""
        return self._index_length

    @index_length.setter
    def index_length(self, value):
        self._index_length = value

    @property
    def index_arrays(self):
        """Arrays of native CS grid coordinates."""
        return self._index_arrays

    @index_arrays.setter
    def index_arrays(self, value):
        self._index_arrays = value

    @property
    def coefficient_space_signature(self):
        """Return a signature for CS coefficient compatibility."""
        return ("CS", int(self.N))

    @property
    def native_grid(self):
        """Return the native CS cell centers as a ``Grid``."""
        if not hasattr(self, "_native_grid"):
            if not hasattr(self, "arr_theta") or not hasattr(self, "arr_phi"):
                raise ValueError("GlobalCSBasis native_grid requires an initialized grid.")
            from kompe.grid import Grid

            self._native_grid = Grid(
                theta=self.arr_theta, phi=self.arr_phi, area_weights=self.unit_area
            )
        return self._native_grid

    @staticmethod
    def _surface_cache_key(name, grid, *parts):
        """Return a cache key for target-grid surface data."""
        signature = getattr(grid, "signature", None)
        if signature is None:
            return None
        return (name, *parts, signature, bool(use_jax()))

    def _cached_surface_matrix(self, name, grid, build, *parts):
        """Return a cached target-grid matrix when possible."""
        key = self._surface_cache_key(name, grid, *parts)
        if key is None:
            return build()
        cache = self._surface_matrix_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        matrix = build()
        cache[key] = matrix
        if len(cache) > self._surface_cache_size:
            cache.popitem(last=False)
        return matrix

    def _cached_surface_operator(self, name, grid, build, *parts):
        """Return a cached target-grid LinearMap when possible."""
        key = self._surface_cache_key(name, grid, *parts)
        if key is None:
            return build()
        cache = self._surface_operator_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        operator = build()
        cache[key] = operator
        if len(cache) > self._surface_cache_size:
            cache.popitem(last=False)
        return operator

    def get_scalar_evaluation_matrix(self, grid, derivative=None):
        """Return the cached CS scalar evaluation matrix."""
        return self._cached_surface_matrix(
            "scalar_evaluation",
            grid,
            lambda: self.evaluate_on_grid(grid, derivative=derivative),
            derivative,
        )

    def get_scalar_evaluation_operator(self, grid, derivative=None):
        """Return the cached CS scalar evaluation operator."""

        def build():
            if self._is_native_grid(grid):
                if derivative is None:
                    return identity_linear_map((self.index_length,))
                elif derivative in {"theta", "phi"}:
                    matrix = self._get_derivative_bundle()[derivative]
                else:
                    raise ValueError(f'Invalid derivative "{derivative}".')
                return as_linear_map(
                    matrix, input_shape=(self.index_length,), output_shape=(self.index_length,)
                )

            if derivative is None:
                return self.scalar_grid_remap_operator(self.native_grid, grid)

            matrix = self.get_scalar_evaluation_matrix(grid, derivative=derivative)
            return as_linear_map(
                matrix, input_shape=(self.index_length,), output_shape=matrix.shape[:-1]
            )

        return self._cached_surface_operator("scalar_evaluation", grid, build, derivative)

    @property
    def scalar_mean_weights(self):
        """Return area-normalized weights for scalar surface means."""
        if not hasattr(self, "unit_area"):
            raise ValueError("GlobalCSBasis scalar mean weights require an initialized grid.")
        weights = np.asarray(self.unit_area, dtype=float)
        total_area = float(np.sum(weights))
        if total_area <= 0.0:
            raise ValueError("GlobalCSBasis unit_area must have positive total area.")
        return weights / total_area

    def scalar_mean(self, coeffs):
        """Return the area-weighted mean of scalar CS coefficients."""
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        if values.shape[-1] != self.index_length:
            raise ValueError(
                "CS scalar coefficients must have the basis index_length on the last axis."
            )
        return xp.tensordot(values, xp.asarray(self.scalar_mean_weights), axes=([-1], [0]))

    def project_scalar_mean_free(self, coeffs):
        """Project scalar CS coefficients to area-weighted zero mean."""
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        mean = self.scalar_mean(values)
        return values - xp.expand_dims(mean, axis=-1)

    def project_helmholtz_mean_free(self, coeffs):
        """Project both CS Helmholtz potentials to zero mean."""
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        if values.shape[-1] == self.index_length:
            return self.project_scalar_mean_free(values)
        if values.shape[-1] == 2 * self.index_length:
            original_shape = values.shape
            reshaped = values.reshape(original_shape[:-1] + (2, self.index_length))
            return self.project_scalar_mean_free(reshaped).reshape(original_shape)
        raise ValueError("CS Helmholtz coefficients must end with index_length or 2*index_length.")

    def _is_native_grid(self, grid):
        """Return whether ``grid`` matches this basis' native points."""
        if grid is self:
            return True
        same_as = getattr(grid, "same_as", None)
        if callable(same_as):
            return bool(same_as(self.native_grid))
        if not hasattr(grid, "theta") or not hasattr(grid, "phi"):
            return False
        from kompe.grid import Grid

        grid_hash = Grid.coordinate_hash(to_numpy(grid.theta), to_numpy(grid.phi))
        return grid_hash == self.native_grid.hash

    def scalar_grid_remap_operator(self, source_grid, target_grid):
        """Return a cached scalar grid-remap operator."""
        return self._grid_remapper.scalar_grid_remap_operator(source_grid, target_grid)

    def tangential_grid_remap_operator(self, source_grid, target_grid):
        """Return a cached tangential grid-remap operator."""
        return self._grid_remapper.tangential_grid_remap_operator(source_grid, target_grid)

    @staticmethod
    def _safe_sin_theta(theta_deg):
        """Return sin(theta) with a pole-safe floor."""
        sin_theta = np.sin(np.deg2rad(np.asarray(theta_deg).reshape(-1)))
        return np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)

    def _coordinate_derivatives(self):
        """Return derivatives of xi/eta with respect to theta/phi."""
        xi, eta, r, block = np.broadcast_arrays(self.arr_xi, self.arr_eta, 1.0, self.arr_block)
        xi, eta, r, block = map(np.ravel, [xi, eta, r, block])

        pc = self.get_Pc(xi, eta, r=r, block=block)
        _, theta, phi = self.cube2spherical(xi, eta, r=r, block=block)

        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        dx_dtheta = r * cos_theta * cos_phi
        dy_dtheta = r * cos_theta * sin_phi
        dz_dtheta = -r * sin_theta
        dx_dphi = -r * sin_theta * sin_phi
        dy_dphi = r * sin_theta * cos_phi
        dz_dphi = np.zeros_like(r)

        dxi_dtheta = pc[:, 0, 0] * dx_dtheta + pc[:, 0, 1] * dy_dtheta + pc[:, 0, 2] * dz_dtheta
        dxi_dphi = pc[:, 0, 0] * dx_dphi + pc[:, 0, 1] * dy_dphi + pc[:, 0, 2] * dz_dphi
        deta_dtheta = pc[:, 1, 0] * dx_dtheta + pc[:, 1, 1] * dy_dtheta + pc[:, 1, 2] * dz_dtheta
        deta_dphi = pc[:, 1, 0] * dx_dphi + pc[:, 1, 1] * dy_dphi + pc[:, 1, 2] * dz_dphi

        return dxi_dtheta, dxi_dphi, deta_dtheta, deta_dphi

    def _get_derivative_bundle(self):
        """Build native-grid angular derivative operators."""
        if self._derivative_bundle is None:
            dxi, deta = self.get_Diff(self.N, coordinate="both", Ns=1, Ni=4, order=1)
            dxi_dtheta, dxi_dphi, deta_dtheta, deta_dphi = self._coordinate_derivatives()

            dtheta = sp.diags(dxi_dtheta) @ dxi + sp.diags(deta_dtheta) @ deta
            dphi_unscaled = sp.diags(dxi_dphi) @ dxi + sp.diags(deta_dphi) @ deta
            sin_theta = self._safe_sin_theta(self.arr_theta)

            # ``phi_unscaled`` is d/dphi. ``phi`` is the azimuthal
            # surface component sin(theta)^-1 d/dphi used by gradients.
            self._derivative_bundle = {
                "theta": dtheta.tocsr(),
                "phi_unscaled": dphi_unscaled.tocsr(),
                "phi": (sp.diags(1.0 / sin_theta) @ dphi_unscaled).tocsr(),
                "sin_theta": sp.diags(sin_theta).tocsr(),
                "inv_sin_theta": sp.diags(1.0 / sin_theta).tocsr(),
                "inv_sin2_theta": sp.diags(1.0 / (sin_theta**2)).tocsr(),
            }
        return self._derivative_bundle

    def evaluate_on_grid(self, grid, derivative=None):
        """Evaluate CS nodal basis or derivatives."""
        xp = get_array_module(getattr(grid, "theta", None), getattr(grid, "phi", None))
        native_grid = self._is_native_grid(grid)
        if not native_grid and derivative is not None:
            raise NotImplementedError(
                "GlobalCSBasis derivative evaluation is currently implemented only "
                "on the native cubed-sphere grid."
            )
        if derivative is None:
            matrix = (
                sp.eye(self.index_length, format="csr")
                if native_grid
                else self._scalar_interpolation_matrix(grid)
            )
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
        elif derivative in {"theta", "phi"}:
            matrix = self._get_derivative_bundle()[derivative].toarray()
        else:
            raise ValueError(f'Invalid derivative "{derivative}".')

        return xp.asarray(matrix)

    def _grid_to_cs_indices(self, grid):
        """Return CS face and cell-center indices."""
        xi, eta, block = self.geo2cube(grid.phi, 90 - grid.theta)
        h = self.xi(1, self.N) - self.xi(0, self.N)
        i = xi / h + (self.N - 1) / 2
        j = eta / h + (self.N - 1) / 2
        return block.reshape(-1), i.reshape(-1), j.reshape(-1)

    def _scalar_interpolation_matrix(self, grid):
        """Return the built-in scalar interpolation as a matrix."""
        return self.interpolate_scalar(
            np.eye(self.index_length), self.arr_theta, self.arr_phi, grid.theta, grid.phi
        )

    def _interpolate_tangential_operator(self, tangential_operator, grid):
        """Interpolate native-grid tangential operators to ``grid``."""
        tangential_operator = np.asarray(tangential_operator)
        east, north, _ = self.interpolate_vector_components(
            tangential_operator[1],
            -tangential_operator[0],
            np.zeros_like(tangential_operator[0]),
            self.arr_theta,
            self.arr_phi,
            grid.theta,
            grid.phi,
        )
        return np.stack([-north, east], axis=0)

    def get_surface_gradient_matrix(self, grid):
        """Return the CS surface-gradient matrix on ``grid``."""

        def build():
            if self._is_native_grid(grid):
                return SurfaceOperators.get_surface_gradient_matrix(self, grid)
            native_gradient = SurfaceOperators.get_surface_gradient_matrix(self, self)
            matrix = self._interpolate_tangential_operator(native_gradient, grid)
            xp = get_array_module(getattr(grid, "theta", None), matrix)
            return xp.asarray(matrix)

        return self._cached_surface_matrix("surface_gradient", grid, build)

    def get_surface_gradient_operator(self, grid):
        """Return the CS surface-gradient operator on ``grid``."""

        def build():
            bundle = self._get_derivative_bundle()
            matrix = sp.vstack([bundle["theta"], bundle["phi"]], format="csr")
            native_operator = as_linear_map(
                matrix, input_shape=(self.index_length,), output_shape=(2, self.index_length)
            )
            if self._is_native_grid(grid):
                return native_operator
            return self.tangential_grid_remap_operator(self.native_grid, grid) @ native_operator

        return self._cached_surface_operator("surface_gradient", grid, build)

    def get_rhat_cross_gradient_matrix(self, grid):
        """Return the CS rhat-cross-gradient matrix on ``grid``."""

        def build():
            if self._is_native_grid(grid):
                return SurfaceOperators.get_rhat_cross_gradient_matrix(self, grid)
            native_rxgrad = SurfaceOperators.get_rhat_cross_gradient_matrix(self, self)
            matrix = self._interpolate_tangential_operator(native_rxgrad, grid)
            xp = get_array_module(getattr(grid, "theta", None), matrix)
            return xp.asarray(matrix)

        return self._cached_surface_matrix("rhat_cross_gradient", grid, build)

    def get_rhat_cross_gradient_operator(self, grid):
        """Return the CS rhat-cross-gradient operator on ``grid``."""

        def build():
            bundle = self._get_derivative_bundle()
            matrix = sp.vstack([-bundle["phi"], bundle["theta"]], format="csr")
            native_operator = as_linear_map(
                matrix, input_shape=(self.index_length,), output_shape=(2, self.index_length)
            )
            if self._is_native_grid(grid):
                return native_operator
            return self.tangential_grid_remap_operator(self.native_grid, grid) @ native_operator

        return self._cached_surface_operator("rhat_cross_gradient", grid, build)

    def get_helmholtz_synthesis_matrix(self, grid):
        """Return the CS Helmholtz synthesis tensor on ``grid``."""

        def build():
            if self._is_native_grid(grid):
                return SurfaceOperators.get_helmholtz_synthesis_matrix(self, grid)
            xp = get_array_module(getattr(grid, "theta", None), getattr(grid, "phi", None))
            native_gradient = SurfaceOperators.get_surface_gradient_matrix(self, self)
            native_rxgrad = np.stack([-native_gradient[1], native_gradient[0]], axis=0)
            target_gradient = self._interpolate_tangential_operator(native_gradient, grid)
            target_rxgrad = self._interpolate_tangential_operator(native_rxgrad, grid)
            return xp.stack([-xp.asarray(target_gradient), xp.asarray(target_rxgrad)], axis=2)

        return self._cached_surface_matrix("helmholtz_synthesis", grid, build)

    def get_helmholtz_synthesis_operator(self, grid):
        """Return the CS Helmholtz synthesis operator on ``grid``."""

        def build():
            matrix = self._native_helmholtz_synthesis_matrix()
            native_operator = as_linear_map(
                matrix, input_shape=(2, self.index_length), output_shape=(2, self.index_length)
            )
            if self._is_native_grid(grid):
                return native_operator
            return self.tangential_grid_remap_operator(self.native_grid, grid) @ native_operator

        return self._cached_surface_operator("helmholtz_synthesis", grid, build)

    def _native_helmholtz_synthesis_matrix(self):
        """Return the sparse native-grid Helmholtz synthesis matrix."""
        bundle = self._get_derivative_bundle()
        theta = bundle["theta"]
        phi = bundle["phi"]
        return sp.bmat([[-theta, -phi], [-phi, theta]], format="csr")

    def get_helmholtz_analysis_operator(self, grid, *, sqrt_weights=None):
        """Return sparse constrained native-grid Helmholtz analysis."""
        if not self._is_native_grid(grid):
            return None

        n = self.index_length
        synthesis = self._native_helmholtz_synthesis_matrix()
        normalized_mean = np.sqrt(n) * self.scalar_mean_weights
        gauges = sp.csr_matrix(
            np.vstack(
                [
                    np.concatenate([normalized_mean, np.zeros(n)]),
                    np.concatenate([np.zeros(n), normalized_mean]),
                ]
            )
        )
        return sparse_constrained_least_squares_map(
            synthesis, gauges, sqrt_weights=sqrt_weights, input_shape=(2, n), output_shape=(2, n)
        )

    def _sparse_laplacian_matrix(self, r=1.0):
        """Return the cached sparse discrete scalar Laplacian."""
        key = float(r)
        if key not in self._laplacian_sparse_cache:
            bundle = self._get_derivative_bundle()
            term_theta = (
                bundle["inv_sin_theta"] @ bundle["theta"] @ bundle["sin_theta"] @ bundle["theta"]
            )
            term_phi = bundle["inv_sin2_theta"] @ bundle["phi_unscaled"] @ bundle["phi_unscaled"]
            self._laplacian_sparse_cache[key] = ((term_theta + term_phi) / (r**2)).tocsr()
        return self._laplacian_sparse_cache[key]

    def laplacian(self, r=1.0):
        """Return the discrete scalar Laplacian matrix."""
        key = float(r)
        if key not in self._laplacian_cache:
            self._laplacian_cache[key] = self._sparse_laplacian_matrix(r).toarray()
        return get_array_module().asarray(self._laplacian_cache[key])

    def get_surface_laplacian_operator(self, r=1.0):
        """Return the native sparse scalar Laplacian operator."""
        return as_linear_map(
            self._sparse_laplacian_matrix(r),
            input_shape=(self.index_length,),
            output_shape=(self.index_length,),
        )

    def get_mean_free_surface_poisson_operator(self, r=1.0):
        """Return the mean-zero inverse of the discrete Laplacian."""
        n = self.index_length
        normalized_mean = np.sqrt(n) * self.scalar_mean_weights
        gauge = sp.csr_matrix(normalized_mean.reshape(1, n))
        return sparse_constrained_least_squares_map(
            self._sparse_laplacian_matrix(r), gauge, input_shape=(n,), output_shape=(n,)
        )

    def get_gridpoints(self, N, flat=False):
        """Generate grid-line indices for a given resolution.

        Parameters
        ----------
        N : int
            Number of grid cells per edge.
        flat : bool, optional
            Whether to return flattened arrays.

        Returns
        -------
        k : ndarray
            Block indices (0-5).
        i : ndarray
            Xi direction indices (0 to N).
        j : ndarray
            Eta direction indices (0 to N).

        Notes
        -----
        Arrays have shape (6,N+1,N+1) if `flat` is ``False``, or
        (6*(N+1)*(N+1),) if `flat` is ``True``.
        Native GlobalCSBasis coefficients are cell-centered at
        ``i + 0.5, j + 0.5`` for ``i, j = 0, ..., N-1``.
        """
        k, i, j = np.meshgrid(np.arange(6), np.arange(N + 1), np.arange(N + 1), indexing="ij")
        if flat:
            return k.reshape(-1), i.reshape(-1), j.reshape(-1)
        else:
            return k, i, j

    def xi(self, i, N):
        """Calculate xi coordinate for grid index.

        Maps index i=0 to -π/4 and i=N to π/4, providing the xi
        coordinate in the cubed sphere grid system.

        Parameters
        ----------
        i : array-like
            Index values (can be non-integer).
        N : int
            Grid resolution (number of cells per edge).

        Returns
        -------
        ndarray
            Xi coordinates in radians from -π/4 to π/4.

        Raises
        ------
        TypeError
            If `N` is not an integer.
        ValueError
            If `N` is less than 1.
        """
        return cs_coordinates.coordinate(i, N)

    def eta(self, j, N):
        """Calculate eta coordinate for grid index.

        Maps index ``j=0`` to -π/4 and ``j=N`` to π/4, providing the eta
        coordinate in the cubed sphere grid system. This function is
        mathematically identical to xi() but is provided separately for
        code clarity.

        Parameters
        ----------
        j : array-like
            Index values (can be non-integer).
        N : int
            Grid resolution (number of cells per edge).

        Returns
        -------
        ndarray
            Eta coordinates in radians from -π/4 to π/4.

        Raises
        ------
        TypeError
            If `N` is not an integer.
        ValueError
            If `N` is less than 1.
        """
        return cs_coordinates.coordinate(j, N)

    def get_delta(self, xi, eta):
        """Calculate delta parameter for metric calculations.

        Computes ``δ = 1 + tan²(ξ) + tan²(η)``.

        Parameters
        ----------
        xi : array-like
            Xi coordinates in radians.
        eta : array-like
            Eta coordinates in radians.

        Returns
        -------
        ndarray
            Delta values with shape determined by broadcasting rules.
        """
        return cs_coordinates.delta(xi, eta)

    def get_metric_tensor(self, xi, eta, r=1, covariant=True):
        """Calculate metric tensor components.

        Calculates the metric tensor components for the cubed sphere
        grid system at given points, which relate coordinate
        differentials to distances according to the equation
        ``ds² = gᵢⱼ dxⁱdxʲ``. Implementation based on equation (12) from
        Yin et al. (2017).

        Parameters
        ----------
        xi : array-like
            Xi coordinates in radians.
        eta : array-like
            Eta coordinates in radians.
        r : array-like, optional
            Radial coordinates.
        covariant : bool, optional
            If ``True`` return covariant components, otherwise return
            contravariant components.

        Returns
        -------
        g : ndarray
            Metric tensor components with shape (N,3,3) where N is the
            number of input points. Last two dimensions are tensor
            indices.
        """
        return cs_coordinates.metric_tensor(xi, eta, r=r, covariant=covariant)

    def cube2cartesian(self, xi, eta, r=1, block=0):
        """Calculate Cartesian ECEF coordinates of given points.

        Output will have same unit as `r`.

        Calculations based on equations from Appendix A of Yin et al.
        (2017).

        Parameters
        ----------
        xi : array-like
            Array of xi coordinates in radians.
        eta : array-like
            Array of eta coordinates in radians.
        r : array-like, optional
            Array of radii.
        block : array-like, optional
            Array of block indices.

        Returns
        -------
        x : array
            Array of Cartesian x coordinates, shape determined by input
            according to broadcasting rules.
        y : array
            Array of Cartesian y coordinates, shape determined by input
            according to broadcasting rules.
        z : array
            Array of Cartesian z coordinates, shape determined by input
            according to broadcasting rules.
        """
        return cs_coordinates.cube_to_cartesian(xi, eta, r=r, block=block)

    def cube2spherical(self, xi, eta, block, r=1, deg=False):
        """Convert from cubed sphere to spherical coordinates.

        Converts cubed sphere coordinates to spherical coordinates
        through intermediate Cartesian coordinates using equations from
        Appendix A of Yin et al. (2017).

        Parameters
        ----------
        xi : array-like
            Xi coordinates in radians.
        eta : array-like
            Eta coordinates in radians.
        block : array-like
            Block indices (0-5)
        r : float or array-like, optional
            Radial coordinates.
        deg : bool, optional
            Return angles in degrees if True, otherwise radians.

        Returns
        -------
        r : ndarray
            Radial coordinates (same units as input r).
        theta : ndarray
            Colatitude in radians or degrees.
        phi : ndarray
            Longitude in radians or degrees.
        """
        return cs_coordinates.cube_to_spherical(xi, eta, block, r=r, deg=deg)

    def get_Pc(self, xi, eta, r=1, block=0, inverse=False):
        """Get Pc matrix.

        Calculates elements of transformation matrix `Pc` at all input
        points.

        The `Pc` matrix transforms Cartesian components ``(ux, uy, uz)``
        to contravariant components in a cubed sphere coordinate
        system::

            |u1| = |P00 P01 P02| |ux|
            |u2| = |P10 P11 P12| |uy|
            |u3| = |P20 P21 P22| |uz|

        The output, `Pc`, will have shape ``(N, 3, 3)``.

        Calculations based on equations from Appendix A of Yin et al.
        (2017), with similar notation.

        Parameters
        ----------
        xi : array-like
            Array of xi coordinates, in radians.
        eta : array-like
            Array of eta coordinates, in radians.
        r : array-like, optional
            Array of radii.
        block : array-like, optional
            Array of block indices.
        inverse : bool, optional
            Set to ``True`` if you want the inverse transformation
            matrix.

        Returns
        -------
        Pc : array
            Transformation matrices `Pc`, one for each point described
            by the input parameters (using broadcasting rules). For
            ``N`` such points, `Pc` will have shape ``(N, 3, 3)``, where
            the last two dimensions refer to column and row of the
            matrix.
        """
        return cs_vectors.pc(xi, eta, r=r, block=block, inverse=inverse)

    def get_Ps(self, xi, eta, r=1, block=0, inverse=False):
        """Get Ps matrix.

        Calculates elements of transformation matrix `Ps` at all input
        points.

        The `Ps` matrix transforms vector components
        ``(u_east, u_north, u_r)`` to contravariant components in a
        cubed sphere coordinate system::

            |u1| = |P00 P01 P02| |u_east|
            |u2| = |P10 P11 P12| |u_north|
            |u3| = |P20 P21 P22| |u_r|

        The output, `Ps`, will have shape ``(N, 3, 3)``.

        Calculations based on equations from Appendix A of Yin et al.
        (2017), with similar notation, except that ``lambda`` and
        ``phi`` is replaced with ``east`` and ``north`` (here, ``phi``
        means longitude, and not latitude as in Yin et al. (2017).

        Parameters
        ----------
        xi : array-like
            Array of xi coordinates, in radians.
        eta : array-like
            Array of eta coordinates, in radians.
        r : array-like, optional
            Array of radii.
        block : array-like, optional
            Array of block indices.
        inverse : bool, optional
            Set to ``True`` if you want the inverse transformation
            matrix.

        Returns
        -------
        Ps : array
            Transformation matrices `Ps`, one for each point described
            by the input parameters (using broadcasting rules). For
            ``N`` such points, `Ps` will have shape ``(N, 3, 3)``, where
            the last two dimensions refer to column and row of the
            matrix.
        """
        return cs_vectors.ps(xi, eta, r=r, block=block, inverse=inverse)

    def get_Qij(self, xi, eta, block_i, block_j):
        """Get Qij matrix.

        Calculates matrix `Qij` that transforms contravariant vector
        components from block `block_i` to `block_j`.

        Calculations are done via transformation to spherical
        coordinates, as suggested by Yin et al. (2017) See equations
        (66) and (67) in their paper.

        It works like this, where ``(u1, u2, u3)`` refer to
        contravariant vector components in the cubed sphere coordinate
        system::

            |u1_j|      |u1_i|
            |u2_j| = Qij|u2_i|
            |u3_j|      |u3_i|

        Parameters
        ----------
        xi : array-like
            Array of xi coordinates on block given by `block_i`, in
            radians.
        eta : array-like
            Array of eta coordinates on block given by `block_i`, in
            radians.
        block_i : array-like, optional
            Indices of block(s) from which to transform vector
            components.
        block_j : array-like, optional
            Indices of block(s) to which to transform vector components.

        Returns
        -------
        Qij : array
            Transformation matrices `Qij`, one for each point described
            by the input parameters (using broadcasting rules). For
            ``N`` such points, `Qij` will have shape ``(N, 3, 3)``,
            where the last two dimensions refer to column and row of the
            matrix.
        """
        return cs_vectors.q_between_blocks(xi, eta, block_i, block_j)

    def get_Q(self, lat, r, inverse=False):
        """Get Q matrix.

        Calculates the matrices that convert from unnormalized spherical
        components to normalized spherical vector components::

            |u_east_normalized |    |u_east |
            |u_north_normalized| = Q|u_north|
            |u_r_normalized    |    |u_r    |

        Based on equations after (A25) in Yin et al. (2017).

        Parameters
        ----------
        lat : array
            Array of latitudes, in degrees.
        r : array
            Array of radii.
        inverse : bool, optional
            Set to ``True`` if you want the inverse transformation
            matrix.

        Returns
        -------
        Q : array
            ``(N, 3, 3)`` array, where ``N`` is the size implied by
            broadcasting the input.
        """
        return cs_vectors.spherical_q(lat, r, inverse=inverse)

    def get_Diff(self, N, coordinate="xi", Ns=1, Ni=4, order=1):
        """Get scalar field differentiation matrix.

        Calculate matrix that differentiates a scalar field, defined on
        a ``(6, N, N)`` grid, with respect to ``xi`` or ``eta``.

        Parameters
        ----------
        N : int
            Number of grid cells in each dimension on each block.
        coordinate : string, {'xi', 'eta', 'both'}
            Which coordinate to differentiate with respect to.
        Ns : int, optional
            Differentiation stencil size.
        Ni : int, optional
            Number of points to use for interpolation for points in the
            stencil that fall on non-integer grid points on neighboring
            blocks.
        order : int, optional
            Order of differentiation. Make sure that ``Ns >= order``.
            Currently only first order differentiation is supported.

        Returns
        -------
        D : sparse matrix
            Sparse ``(6*N*N, 6*N*N)`` matrix that calculates the
            derivative of a scalar field with respect to ``xi`` or
            ``eta`` as ``derivative = D.dot(f)``, where ``f`` is the
            scalar field.

        Raises
        ------
        ValueError
            If ``coordinate`` is not ``'xi'``, ``'eta'``, or ``'both'``.
            If ``Ns`` is less than ``order``.
        NotImplementedError
            If ``order`` is not 1.
        """
        return self._finite_differences.difference_matrix(
            N, coordinate=coordinate, Ns=Ns, Ni=Ni, order=order
        )

    def get_interpolation_matrix(self, k, i, j, N, Ni, weights=None, rows=None):
        """Get matrix for grid to cubed sphere interpolation.

        Calculates a sparse matrix D that interpolates from grid points
        in a ``(6, N, N)`` grid to the indices (`k`, `i`, `j`).

        `D` will have ``6*N**2`` columns that refer to the ``(6, N, N)``
        grid points, spanning the 6 blocks in the cubed sphere, with
        duplicate points on the boundaries.

        Parameters
        ----------
        k : array-like
            Integer indices that refer to cube block. Must be ``>= 0``
            and ``<= 5``. Will be flattened.
        i : array-like
            Integer indices that refer to the ``xi``-direction (but can
            be negative or ``>= N``). Will be flattened.
        j : array-like
            Integer indices that refer to the ``eta``-direction (but can
            be negative or ``>= N``). Will be flattened.
        N : int
            Number of grid points.
        Ni : int
            Number of interpolation points. Must be ``<= N`` (4 is often
            appropriate).
        weights : array-like, optional
            If different values of `k`, `i`, `j` are assigned to the
            same row, the corresponding element will have value 1 (or
            whatever the interpolation dictates) unless weights is
            specified. For differentiation, use weights to specify the
            stencil coefficients.
        rows : array-like, optional
            The row index of each element in `k`, `i`, `j`. Different
            elements of `k`, `i`, `j` can be put in the same row. If not
            specified, each element in `k`, `i`, `j` will be given its
            own row.

        Returns
        -------
        D : sparse matrix
            ``(rows.max() + 1 by 6*N*N)`` matrix that, when multiplied
            by a vector containing a scalar field on the ``6*N*N`` grid
            points, produces interpolated values at the given grid
            points. The grid points may be outside the cube blocks, for
            example they can be negative (actually that's the point,
            otherwise this function would not be needed).
        """
        return self._finite_differences.interpolation_matrix(
            k, i, j, N, Ni, weights=weights, rows=rows
        )

    def block(self, lon, lat):
        """Determine cube faces (blocks) of spherical coordinates.

        For each input point, determines which of the six cube faces is
        closest by calculating distances to face midpoints in Cartesian
        space.

        Parameters
        ----------
        lon : array-like
            Geocentric longitude(s) in degrees.
        lat : array-like
            Geocentric latitude(s) in degrees.

        Returns
        -------
        ndarray
            Indices of the block that each (lon, lat) point belongs to:
            - 0 (I)   : Equatorial face at 0° longitude
            - 1 (II)  : Equatorial face at 90° longitude
            - 2 (III) : Equatorial face at 180° longitude
            - 3 (IV)  : Equatorial face at 270° longitude
            - 4 (V)   : North polar face
            - 5 (VI)  : South polar face

        Notes
        -----
        The method uses Euclidean distances to face midpoints in
        Cartesian space to determine block membership. This ensures
        unique block assignment even for points near block boundaries.
        """
        return cs_coordinates.cube_face(lon, lat)

    def geo2cube(self, lon, lat, block=None):
        """Convert geocentric coordinates to cube coordinates.

        Input parameters must have same shape. Output will have same
        shape.

        Parameters
        ----------
        lon : array
            Geocentric longitude(s) to convert to cube coords, in
            degrees.
        lat : array
            Geocentric latitude(s) to convert to cube coords, in
            degrees.
        block : array-like, optional
            Option to specify cube block. If ``None``, it will be
            calculated. If specified, be careful because the function
            will map points at opposite side of the sphere to specified
            block.

        Returns
        -------
        xi : array
            `xi`, as defined in Ronchi et al. (1996). Unit is radians.
        eta : array
            `eta`, as defined in Ronchi et al. (1996). Unit is radians.
        block : array
            Index of the block that `xi`, `eta` belongs to.
        """
        return cs_coordinates.geo_to_cube(lon, lat, block=block)

    def interpolate_vector_components(
        self, u_east, u_north, u_r, theta, phi, theta_target, phi_target, **kwargs
    ):
        """Interpolate vector components.

        Interpolates vector components defined on (theta, phi) to given
        spherical coordinates. Extra trailing dimensions on the
        component arrays are treated as independent vector fields and
        interpolated in one call.

        Broadcasting rules apply for input and output separately.

        Parameters
        ----------
        u_east : array
            Array of eastward components.
        u_north : array
            Array of northward components.
        u_r : array
            Array of radial components.
        theta : array
            Array of coordinates for components.
        phi : array
            Array of coordinates for vector components.
        theta_target : array
            Array of target coordinates.
        phi_target : array
            Array of target coordinates.

        **kwargs
            Passed to scipy.interpolate.griddata which performs the
            interpolation on each block.

        Returns
        -------
        interpolated_vector : array
            3 x N vector of interpolated components (east, north, up).
        """
        return self._grid_remapper.interpolate_vector_components(
            u_east, u_north, u_r, theta, phi, theta_target, phi_target, **kwargs
        )

    def interpolate_scalar(self, scalar, theta, phi, theta_target, phi_target, **kwargs):
        """Interpolate scalar values.

        Interpolate scalar values defined on (`theta`, `phi`) to given
        spherical coordinates.  Extra trailing dimensions on ``scalar``
        are treated as independent scalar fields and interpolated in
        one call.

        Broadcasting rules apply for input and output separately.

        Parameters
        ----------
        scalar : array
            Array of scalar values.
        theta : array
            Array of coordinates for components.
        phi : array
            Array of coordinates for vector components.
        theta_target : array
            Array of target coordinates.
        phi_target : array
            Array of target coordinates.

        **kwargs
            Passed to scipy.interpolate.griddata which performs the
            interpolation on each block.

        Returns
        -------
        interpolated_scalar : array
            Array of interpolated components (east, north, up).
        """
        return self._grid_remapper.interpolate_scalar(
            scalar, theta, phi, theta_target, phi_target, **kwargs
        )


__all__ = ["GlobalCSBasis"]
