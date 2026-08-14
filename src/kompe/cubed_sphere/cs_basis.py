"""Global cubed-sphere surface basis."""

from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

from kompe.core import SurfaceDifferentialBasis
from kompe.cubed_sphere.cs_differencing import CSFiniteDifferences
from kompe.cubed_sphere.cs_grid import CSGridRemapper, GlobalCSMesh
from kompe.math import as_linear_map, identity_linear_map
from kompe.math.backend import get_array_module, to_numpy, use_jax
from kompe.math.least_squares_solver import sparse_constrained_least_squares_map


class GlobalCSBasis(SurfaceDifferentialBasis):
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
    cells_per_face : int
        Number of grid cells along each cube-face edge.
    xi : ndarray
        Xi coordinates of native cell centers, in radians.
    eta : ndarray
        Eta coordinates of native cell centers, in radians.
    theta : ndarray
        Colatitude coordinates of native cell centers, in degrees.
    phi : ndarray
        Longitude coordinates of native cell centers, in degrees.
    face : ndarray
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

    def __init__(self, cells_per_face):
        """Initialize the cubed sphere basis.

        Initialize arrays for a grid with the requested number of cells along
        each cube-face edge.

        Parameters
        ----------
        cells_per_face : int
            Number of grid cells per cube edge. Must be even.

        Raises
        ------
        TypeError
            If ``cells_per_face`` is not an integer.
        ValueError
            If ``cells_per_face`` is not a positive even number.
        """
        self.kind = "CS"
        self._derivative_bundle = None
        self._laplacian_cache = {}
        self._laplacian_sparse_cache = {}
        self._grid_remapper = CSGridRemapper(self)
        self._finite_differences = CSFiniteDifferences(self)
        self._surface_matrix_cache = OrderedDict()
        self._surface_operator_cache = OrderedDict()

        if isinstance(cells_per_face, bool) or not isinstance(cells_per_face, (int, np.integer)):
            raise TypeError("cells_per_face must be an integer")
        if cells_per_face <= 0:
            raise ValueError("Cubed sphere grid dimension must be positive")
        if cells_per_face % 2 != 0:
            raise ValueError("Cubed sphere grid dimension must be even")

        self.cells_per_face = int(cells_per_face)
        self.mesh = GlobalCSMesh(self.cells_per_face)

        self.index_names = ("theta", "phi")
        self.index_length = self.mesh.size
        self.index_arrays = (self.mesh.theta, self.mesh.phi)

        self.validate_metadata()

    def __repr__(self):
        """Summarize the global cubed-sphere coefficient space."""
        return (
            f"GlobalCSBasis(cells_per_face={self.cells_per_face}, "
            f"index_length={self.index_length})"
        )

    def clear_cache(self, *, shared_remaps=False):
        """Clear derived operators and matrices owned by this basis.

        Set ``shared_remaps`` to also clear the bounded process-wide cache of
        geometry-only interpolation matrices.
        """
        self._derivative_bundle = None
        self._laplacian_cache.clear()
        self._laplacian_sparse_cache.clear()
        self._surface_matrix_cache.clear()
        self._surface_operator_cache.clear()
        self._grid_remapper.clear_cache()
        if shared_remaps:
            self._grid_remapper.clear_shared_cache()

    def cache_info(self):
        """Return cache occupancy without exposing mutable cache objects."""
        return {
            "derivatives_built": self._derivative_bundle is not None,
            "laplacian_matrices": len(self._laplacian_cache),
            "sparse_laplacian_matrices": len(self._laplacian_sparse_cache),
            "surface_matrices": len(self._surface_matrix_cache),
            "surface_operators": len(self._surface_operator_cache),
            "surface_max_size": self._surface_cache_size,
            "remap_operators": self._grid_remapper.cache_info(),
            "shared_remap_matrices": self._grid_remapper.shared_cache_info(),
        }

    @property
    def coefficient_space_signature(self):
        """Return a signature for CS coefficient compatibility."""
        return ("CS", int(self.cells_per_face))

    @property
    def native_grid(self):
        """Return the native CS cell centers as a ``SphericalGrid``."""
        return self.mesh.cell_centers

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

    def scalar_evaluation_matrix(self, grid, derivative=None):
        """Return the cached CS scalar evaluation matrix."""
        return self._cached_surface_matrix(
            "scalar_evaluation",
            grid,
            lambda: self._build_scalar_evaluation_matrix(grid, derivative=derivative),
            derivative,
        )

    def scalar_evaluation_operator(self, grid, derivative=None):
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

            matrix = self.scalar_evaluation_matrix(grid, derivative=derivative)
            return as_linear_map(
                matrix, input_shape=(self.index_length,), output_shape=matrix.shape[:-1]
            )

        return self._cached_surface_operator("scalar_evaluation", grid, build, derivative)

    @property
    def scalar_mean_weights(self):
        """Return area-normalized weights for scalar surface means."""
        weights = np.asarray(self.mesh.cell_areas.reshape(-1), dtype=float)
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
        from kompe.grid import SphericalGrid

        if isinstance(grid, SphericalGrid):
            return grid.same_as(self.native_grid)
        if not hasattr(grid, "theta") or not hasattr(grid, "phi"):
            return False
        grid = SphericalGrid(theta=to_numpy(grid.theta), phi=to_numpy(grid.phi))
        return grid.same_as(self.native_grid)

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
        xi, eta, r, block = np.broadcast_arrays(self.mesh.xi, self.mesh.eta, 1.0, self.mesh.face)
        xi, eta, r, block = map(np.ravel, [xi, eta, r, block])

        pc = self.mesh.projection.cartesian_to_cube_vector_matrix(xi, eta, radius=r, face=block)
        _, theta, phi = self.mesh.projection.cube_to_spherical(xi, eta, block, radius=r)

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
            dxi, deta = self._finite_differences.difference_matrix(
                self.cells_per_face,
                coordinate="both",
                Ns=1,
                Ni=4,
                order=1,
            )
            dxi_dtheta, dxi_dphi, deta_dtheta, deta_dphi = self._coordinate_derivatives()

            dtheta = sp.diags(dxi_dtheta) @ dxi + sp.diags(deta_dtheta) @ deta
            dphi_unscaled = sp.diags(dxi_dphi) @ dxi + sp.diags(deta_dphi) @ deta
            sin_theta = self._safe_sin_theta(self.mesh.theta)

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

    def _build_scalar_evaluation_matrix(self, grid, derivative=None):
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
                else self.interpolate_scalar(
                    np.eye(self.index_length),
                    self.mesh.theta,
                    self.mesh.phi,
                    grid.theta,
                    grid.phi,
                )
            )
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
        elif derivative in {"theta", "phi"}:
            matrix = self._get_derivative_bundle()[derivative].toarray()
        else:
            raise ValueError(f'Invalid derivative "{derivative}".')

        return xp.asarray(matrix)

    def _interpolate_tangential_operator(self, tangential_operator, grid):
        """Interpolate native-grid tangential operators to ``grid``."""
        tangential_operator = np.asarray(tangential_operator)
        theta, phi, _ = self.interpolate_vector(
            tangential_operator[0],
            tangential_operator[1],
            np.zeros_like(tangential_operator[0]),
            self.mesh.theta,
            self.mesh.phi,
            grid.theta,
            grid.phi,
        )
        return np.stack([theta, phi], axis=0)

    def surface_gradient_matrix(self, grid):
        """Return the CS surface-gradient matrix on ``grid``."""

        def build():
            if self._is_native_grid(grid):
                return SurfaceDifferentialBasis.surface_gradient_matrix(self, grid)
            native_gradient = SurfaceDifferentialBasis.surface_gradient_matrix(
                self, self.native_grid
            )
            matrix = self._interpolate_tangential_operator(native_gradient, grid)
            xp = get_array_module(getattr(grid, "theta", None), matrix)
            return xp.asarray(matrix)

        return self._cached_surface_matrix("surface_gradient", grid, build)

    def surface_gradient_operator(self, grid):
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

    def rhat_cross_gradient_matrix(self, grid):
        """Return the CS rhat-cross-gradient matrix on ``grid``."""

        def build():
            if self._is_native_grid(grid):
                return SurfaceDifferentialBasis.rhat_cross_gradient_matrix(self, grid)
            native_rxgrad = SurfaceDifferentialBasis.rhat_cross_gradient_matrix(
                self, self.native_grid
            )
            matrix = self._interpolate_tangential_operator(native_rxgrad, grid)
            xp = get_array_module(getattr(grid, "theta", None), matrix)
            return xp.asarray(matrix)

        return self._cached_surface_matrix("rhat_cross_gradient", grid, build)

    def rhat_cross_gradient_operator(self, grid):
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

    def helmholtz_synthesis_matrix(self, grid):
        """Return the CS Helmholtz synthesis tensor on ``grid``."""

        def build():
            if self._is_native_grid(grid):
                return SurfaceDifferentialBasis.helmholtz_synthesis_matrix(self, grid)
            xp = get_array_module(getattr(grid, "theta", None), getattr(grid, "phi", None))
            native_gradient = SurfaceDifferentialBasis.surface_gradient_matrix(
                self, self.native_grid
            )
            native_rxgrad = np.stack([-native_gradient[1], native_gradient[0]], axis=0)
            target_gradient = self._interpolate_tangential_operator(native_gradient, grid)
            target_rxgrad = self._interpolate_tangential_operator(native_rxgrad, grid)
            return xp.stack([-xp.asarray(target_gradient), xp.asarray(target_rxgrad)], axis=2)

        return self._cached_surface_matrix("helmholtz_synthesis", grid, build)

    def helmholtz_synthesis_operator(self, grid):
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

    def helmholtz_analysis_operator(self, grid, *, sqrt_weights=None):
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

    def _surface_laplacian(self, r=1.0):
        """Return the discrete scalar Laplacian matrix."""
        key = float(r)
        if key not in self._laplacian_cache:
            self._laplacian_cache[key] = self._sparse_laplacian_matrix(r).toarray()
        return get_array_module().asarray(self._laplacian_cache[key])

    def surface_laplacian_operator(self, r=1.0):
        """Return the native sparse scalar Laplacian operator."""
        return as_linear_map(
            self._sparse_laplacian_matrix(r),
            input_shape=(self.index_length,),
            output_shape=(self.index_length,),
        )

    def mean_free_surface_poisson_operator(self, r=1.0):
        """Return the mean-zero inverse of the discrete Laplacian."""
        n = self.index_length
        normalized_mean = np.sqrt(n) * self.scalar_mean_weights
        gauge = sp.csr_matrix(normalized_mean.reshape(1, n))
        return sparse_constrained_least_squares_map(
            self._sparse_laplacian_matrix(r), gauge, input_shape=(n,), output_shape=(n,)
        )

    def interpolate_vector(
        self, u_theta, u_phi, u_radial, theta, phi, theta_target, phi_target, **kwargs
    ):
        """Interpolate canonical spherical vector components.

        Interpolates ``(theta, phi, radial)`` components defined on spherical
        coordinates to target spherical coordinates. Extra trailing dimensions on the
        component arrays are treated as independent vector fields and
        interpolated in one call.

        Broadcasting rules apply for input and output separately.

        Parameters
        ----------
        u_theta : array
            Array of southward components.
        u_phi : array
            Array of eastward components.
        u_radial : array
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
            Tuple of interpolated ``(theta, phi, radial)`` components.
        """
        return self._grid_remapper.interpolate_vector(
            u_theta, u_phi, u_radial, theta, phi, theta_target, phi_target, **kwargs
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
            Interpolated scalar values.
        """
        return self._grid_remapper.interpolate_scalar(
            scalar, theta, phi, theta_target, phi_target, **kwargs
        )


__all__ = ["GlobalCSBasis"]
