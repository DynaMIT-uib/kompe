"""Native differential and inverse operators on a global cubed-sphere mesh."""

from functools import cached_property

import numpy as np
import scipy.sparse as sp

from kompe.cache import BoundedCache
from kompe.cubed_sphere.global_differencing import global_cs_derivative_matrices
from kompe.cubed_sphere.global_mesh import GlobalCSMesh
from kompe.math import as_linear_map
from kompe.math.backend import backend_context, readonly_numpy_array, to_numpy
from kompe.math.least_squares_solver import sparse_least_squares_map


class GlobalCSOperators:
    """Collocated differential operators sharing one immutable mesh.

    Input values are flattened cell-centre samples, not basis coefficients.
    Tangential outputs have shape ``(2, mesh.size)`` in ``(theta, phi)`` order.
    Matrices are assembled once on the CPU; LinearMap application preserves
    NumPy/JAX inputs. The current stencils require an even mesh resolution.
    """

    def __init__(self, mesh):
        if not isinstance(mesh, GlobalCSMesh):
            raise TypeError("mesh must be a GlobalCSMesh.")
        if mesh.cells_per_edge % 2:
            raise ValueError("Global-CS differential operators require an even mesh resolution.")
        self.mesh = mesh
        self._operator_cache = BoundedCache(16)

    @property
    def signature(self):
        """Identify the collocated discretization and its geometry."""
        return ("GLOBAL_CS_OPERATORS", self.mesh.signature)

    def clear_cache(self):
        """Release native matrices and cached operators."""
        for name in (
            "_native_derivatives",
            "_unit_surface_laplacian_matrix",
            "_native_helmholtz_synthesis_matrix",
        ):
            self.__dict__.pop(name, None)
        self._operator_cache.clear()

    def cache_info(self):
        """Summarize native calculation caches."""
        return {
            "derivatives_built": "_native_derivatives" in self.__dict__,
            "laplacian_built": "_unit_surface_laplacian_matrix" in self.__dict__,
            "operators": len(self._operator_cache),
            "max_size": self._operator_cache.max_size,
        }

    @cached_property
    def scalar_mean_weights(self):
        """Area-normalized weights for a mean over cell-centre samples."""
        areas = self.mesh.cell_areas.reshape(-1)
        return readonly_numpy_array(areas / np.sum(areas))

    def _coordinate_derivatives(self):
        """Return derivatives of xi/eta with respect to theta/phi."""
        xi, eta, radius, face = np.broadcast_arrays(
            self.mesh.xi, self.mesh.eta, 1.0, self.mesh.face
        )
        xi, eta, radius, face = map(np.ravel, [xi, eta, radius, face])

        pc = self.mesh.projection.cartesian_to_cube_vector_array(xi, eta, radius=radius, face=face)
        _, theta, phi = self.mesh.projection.cube_to_spherical(xi, eta, face, radius=radius)

        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        dx_dtheta = radius * cos_theta * cos_phi
        dy_dtheta = radius * cos_theta * sin_phi
        dz_dtheta = -radius * sin_theta
        dx_dphi = -radius * sin_theta * sin_phi
        dy_dphi = radius * sin_theta * cos_phi
        dz_dphi = np.zeros_like(radius)

        dxi_dtheta = pc[:, 0, 0] * dx_dtheta + pc[:, 0, 1] * dy_dtheta + pc[:, 0, 2] * dz_dtheta
        dxi_dphi = pc[:, 0, 0] * dx_dphi + pc[:, 0, 1] * dy_dphi + pc[:, 0, 2] * dz_dphi
        deta_dtheta = pc[:, 1, 0] * dx_dtheta + pc[:, 1, 1] * dy_dtheta + pc[:, 1, 2] * dz_dtheta
        deta_dphi = pc[:, 1, 0] * dx_dphi + pc[:, 1, 1] * dy_dphi + pc[:, 1, 2] * dz_dphi

        # These coefficients immediately enter SciPy sparse matrices. Keep
        # that CPU boundary explicit when the active numerical backend is JAX.
        return tuple(to_numpy(values) for values in (dxi_dtheta, dxi_dphi, deta_dtheta, deta_dphi))

    @cached_property
    def _native_derivatives(self):
        """Build native-grid angular derivative operators."""
        with backend_context("numpy"):
            dxi, deta = global_cs_derivative_matrices(
                self.mesh.projection,
                self.mesh.cells_per_edge,
            )
            dxi_dtheta, dxi_dphi, deta_dtheta, deta_dphi = self._coordinate_derivatives()

        dtheta = sp.diags(dxi_dtheta) @ dxi + sp.diags(deta_dtheta) @ deta
        dphi_unscaled = sp.diags(dxi_dphi) @ dxi + sp.diags(deta_dphi) @ deta
        # The required even, cell-centred mesh does not sample either pole.
        sin_theta = np.sin(np.deg2rad(self.mesh.theta))

        # ``phi_unscaled`` is d/dphi. ``phi`` is the azimuthal
        # surface component sin(theta)^-1 d/dphi used by gradients.
        return {
            "theta": dtheta.tocsr(),
            "phi_unscaled": dphi_unscaled.tocsr(),
            "phi": (sp.diags(1.0 / sin_theta) @ dphi_unscaled).tocsr(),
            "sin_theta": sp.diags(sin_theta).tocsr(),
            "inv_sin_theta": sp.diags(1.0 / sin_theta).tocsr(),
            "inv_sin2_theta": sp.diags(1.0 / (sin_theta**2)).tocsr(),
        }

    def surface_gradient_matrices(self):
        """Return sparse unit-sphere gradient components (theta, phi)."""
        return self._native_derivatives["theta"], self._native_derivatives["phi"]

    def surface_gradient_operator(self):
        """Map cell values to their unit-sphere tangential gradient."""

        def build():
            matrix = sp.vstack(self.surface_gradient_matrices(), format="csr")
            return as_linear_map(
                matrix, input_shape=(self.mesh.size,), output_shape=(2, self.mesh.size)
            )

        return self._operator_cache.get_or_create("surface_gradient", build)

    def rhat_cross_gradient_operator(self):
        """Map cell values to r-hat cross their unit-sphere gradient."""

        def build():
            theta, phi = self.surface_gradient_matrices()
            return as_linear_map(
                sp.vstack([-phi, theta], format="csr"),
                input_shape=(self.mesh.size,),
                output_shape=(2, self.mesh.size),
            )

        return self._operator_cache.get_or_create("rhat_cross_gradient", build)

    def helmholtz_synthesis_operator(self):
        """Evaluate -grad(Phi) + r-hat cross grad(Psi) at the cell centres."""
        return self._operator_cache.get_or_create(
            "helmholtz_synthesis",
            lambda: as_linear_map(
                self._native_helmholtz_synthesis_matrix,
                input_shape=(2, self.mesh.size),
                output_shape=(2, self.mesh.size),
            ),
        )

    @cached_property
    def _native_helmholtz_synthesis_matrix(self):
        """Return the sparse native-grid Helmholtz synthesis matrix."""
        derivatives = self._native_derivatives
        theta = derivatives["theta"]
        phi = derivatives["phi"]
        return sp.bmat([[-theta, -phi], [-phi, theta]], format="csr")

    def helmholtz_analysis_operator(self, *, sqrt_weights=None):
        """Return sparse constrained native-grid Helmholtz analysis."""
        n = self.mesh.size
        synthesis = self._native_helmholtz_synthesis_matrix
        mean = self.scalar_mean_weights
        gauges = sp.csr_matrix(
            np.vstack(
                [
                    np.concatenate([mean, np.zeros(n)]),
                    np.concatenate([np.zeros(n), mean]),
                ]
            )
        )
        return sparse_least_squares_map(
            synthesis, gauges, sqrt_weights=sqrt_weights, input_shape=(2, n), output_shape=(2, n)
        )

    @cached_property
    def _unit_surface_laplacian_matrix(self):
        """Return the sparse scalar Laplacian on the unit sphere."""
        derivatives = self._native_derivatives
        term_theta = (
            derivatives["inv_sin_theta"]
            @ derivatives["theta"]
            @ derivatives["sin_theta"]
            @ derivatives["theta"]
        )
        term_phi = (
            derivatives["inv_sin2_theta"]
            @ derivatives["phi_unscaled"]
            @ derivatives["phi_unscaled"]
        )
        return (term_theta + term_phi).tocsr()

    def surface_laplacian_operator(self, r=1.0):
        """Return the native scalar Laplacian, scaled by 1/r².

        These collocated stencils annihilate constants but are not exactly
        area-conservative at finite resolution. Check conservation and
        harmonic eigenvalues by convergence, not finite-volume identities.
        """
        unit = self._operator_cache.get_or_create(
            "surface_laplacian", lambda: as_linear_map(self._unit_surface_laplacian_matrix)
        )
        return unit if r == 1.0 else (1.0 / float(r) ** 2) * unit

    def scalar_smoothness_operator(self):
        """Return R with ||R f||² equal to the discrete mean of |grad_s f|².

        The gradient is on the unit sphere. Assemble its weighted sparse
        rows directly so normal diagonals remain sparse reductions rather
        than coefficient-by-coefficient operator probes.
        """

        def build():
            weights = np.sqrt(self.scalar_mean_weights)[:, None]
            theta, phi = self.surface_gradient_matrices()
            matrix = sp.vstack([theta.multiply(weights), phi.multiply(weights)], format="csr")
            return as_linear_map(
                matrix, input_shape=(self.mesh.size,), output_shape=(2, self.mesh.size)
            )

        return self._operator_cache.get_or_create("scalar_smoothness", build)

    def helmholtz_smoothness_operator(self):
        """Penalize the area mean of div_s(F)² + curl_r(F)² on the unit sphere.

        F = -grad(Phi) + r-hat cross grad(Psi). The weighted residuals are
        -Laplacian(Phi) and Laplacian(Psi), not a vector-gradient norm.
        """

        def build():
            weights = np.sqrt(self.scalar_mean_weights)[:, None]
            laplacian = self._unit_surface_laplacian_matrix.multiply(weights)
            matrix = sp.block_diag([-laplacian, laplacian], format="csr")
            return as_linear_map(
                matrix, input_shape=(2, self.mesh.size), output_shape=(2, self.mesh.size)
            )

        return self._operator_cache.get_or_create("helmholtz_smoothness", build)

    def mean_free_surface_poisson_operator(self, r=1.0):
        """Invert the Laplacian with exact zero area mean, reusing unit-sphere factors."""

        def build():
            n = self.mesh.size
            gauge = sp.csr_matrix(self.scalar_mean_weights.reshape(1, n))
            return sparse_least_squares_map(
                self._unit_surface_laplacian_matrix,
                gauge,
                input_shape=(n,),
                output_shape=(n,),
            )

        unit = self._operator_cache.get_or_create("mean_free_surface_poisson", build)
        return unit if r == 1.0 else float(r) ** 2 * unit


__all__ = ["GlobalCSOperators"]
