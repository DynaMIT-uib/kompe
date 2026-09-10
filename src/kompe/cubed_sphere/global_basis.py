"""Global cubed-sphere surface basis."""

from functools import cached_property

import numpy as np

from kompe.basis import SurfaceDifferentialBasis
from kompe.cache import BoundedCache
from kompe.cubed_sphere.global_mesh import GlobalCSMesh
from kompe.cubed_sphere.global_remapping import GlobalCSRemapper
from kompe.math import as_linear_map, identity_linear_map, take_linear_map
from kompe.math.backend import readonly_numpy_array, to_numpy


class GlobalCSBasis(SurfaceDifferentialBasis):
    """Cell-centred scalar and Helmholtz basis on a global cubed sphere.

    Each face uses local ``(xi, eta)`` coordinates mapped to spherical
    ``(theta, phi)`` coordinates. Scalar coefficients live at cell centres;
    gradients, Helmholtz fields, and Laplacians are evaluated from those
    values. The associated :attr:`mesh` owns coordinates, topology, metric
    values, and cell areas.

    Attributes
    ----------
    cells_per_edge : int
        Number of grid cells along each cube-face edge.
    mesh : GlobalCSMesh
        Native six-face mesh and its spherical geometry.

    Notes
    -----
    The cubed sphere grid is organized into six faces as shown below,
    which defines the face structure of the grid::

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

    Face indices:

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

    def __init__(self, cells_per_edge=None, *, mesh=None):
        """Initialize the cubed sphere basis.

        Initialize arrays for a grid with the requested number of cells along
        each cube-face edge.

        Parameters
        ----------
        cells_per_edge : int, optional
            Number of grid cells per cube edge. Must be even.
        mesh : GlobalCSMesh, optional
            Existing mesh to reuse instead of constructing one. Supply
            either cells_per_edge or mesh, not both.

        Raises
        ------
        TypeError
            If ``cells_per_edge`` is not an integer.
        ValueError
            If ``cells_per_edge`` is not a positive even number.
        """
        self.kind = "CS"
        self.remapper = GlobalCSRemapper()
        self._surface_operator_cache = BoundedCache(16)

        if mesh is None:
            mesh = GlobalCSMesh(cells_per_edge)
        elif cells_per_edge is not None:
            raise ValueError("Supply either cells_per_edge or mesh, not both.")
        elif not isinstance(mesh, GlobalCSMesh):
            raise TypeError("mesh must be a GlobalCSMesh.")
        if mesh.cells_per_edge % 2 != 0:
            raise ValueError("Cubed sphere grid dimension must be even")

        self.cells_per_edge = mesh.cells_per_edge
        self.mesh = mesh

        self.index_names = ("theta", "phi")
        self.coefficient_count = self.mesh.size
        self.index_arrays = (self.mesh.theta, self.mesh.phi)

        self.validate_metadata()

    def __repr__(self):
        """Summarize the global cubed-sphere coefficient space."""
        return (
            f"GlobalCSBasis(cells_per_edge={self.cells_per_edge}, "
            f"coefficient_count={self.coefficient_count})"
        )

    def clear_cache(self, *, shared_remaps=False):
        """Clear target-grid caches and the shared mesh's native operators.

        Set ``shared_remaps`` to also clear the bounded process-wide cache of
        geometry-only interpolation matrices.
        Previously returned LinearMaps remain usable.
        """
        self.mesh.operators.clear_cache()
        self._surface_operator_cache.clear()
        self.remapper.clear_cache()
        if shared_remaps:
            self.remapper.clear_shared_cache()

    def cache_info(self):
        """Return cache occupancy without exposing mutable cache objects."""
        return {
            "mesh_operators": self.mesh.operators.cache_info(),
            "surface_operators": len(self._surface_operator_cache),
            "surface_max_size": self._surface_operator_cache.max_size,
            "remap_operators": self.remapper.cache_info(),
            "shared_remap_matrices": self.remapper.shared_cache_info(),
        }

    @property
    def coefficient_space_signature(self):
        """Return a signature for CS coefficient compatibility."""
        return ("CS", int(self.cells_per_edge))

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
        return (name, *parts, signature)

    def _cached_surface_operator(self, name, grid, build, *parts):
        """Return a cached target-grid LinearMap when possible."""
        key = self._surface_cache_key(name, grid, *parts)
        if key is None:
            return build()
        return self._surface_operator_cache.get_or_create(key, build)

    def scalar_evaluation_array(self, grid, gradient_component=None, *, persist=True):
        """Materialize the canonical CS scalar evaluation operator."""
        return self.scalar_evaluation_operator(
            grid, gradient_component=gradient_component, persist=persist
        ).to_array()

    def scalar_evaluation_operator(self, grid, gradient_component=None, *, persist=True):
        """Return the cached CS scalar evaluation operator."""

        def build():
            if self._is_native_grid(grid):
                if gradient_component is None:
                    return identity_linear_map((self.coefficient_count,))
                elif gradient_component in {"theta", "phi"}:
                    matrix = self.mesh.operators.surface_gradient_matrices()[
                        0 if gradient_component == "theta" else 1
                    ]
                else:
                    raise ValueError(f'Invalid gradient_component "{gradient_component}".')
                return as_linear_map(
                    matrix,
                    input_shape=(self.coefficient_count,),
                    output_shape=(self.coefficient_count,),
                )

            if gradient_component is None:
                return self.remapper.scalar_operator(self.native_grid, grid)
            if gradient_component not in {"theta", "phi"}:
                raise ValueError(f'Invalid gradient_component "{gradient_component}".')
            component = take_linear_map(
                (2, grid.size), 0 if gradient_component == "theta" else 1, axis=0
            )
            return component @ self.surface_gradient_operator(grid)

        return self._cached_surface_operator("scalar_evaluation", grid, build, gradient_component)

    @cached_property
    def scalar_constant_coefficients(self):
        """A unit constant has value one at every native node."""
        return readonly_numpy_array(np.ones(self.coefficient_count))

    @property
    def scalar_mean_weights(self):
        """Return the mesh's area-normalized weights for nodal coefficients."""
        return self.mesh.operators.scalar_mean_weights

    def _is_native_grid(self, grid):
        """Return whether ``grid`` matches this basis' native points."""
        from kompe.grid import SphericalGrid

        if isinstance(grid, SphericalGrid):
            return grid.same_as(self.native_grid)
        if not hasattr(grid, "theta") or not hasattr(grid, "phi"):
            return False
        grid = SphericalGrid(theta=to_numpy(grid.theta), phi=to_numpy(grid.phi))
        return grid.same_as(self.native_grid)

    def surface_gradient_array(self, grid, *, persist=True):
        """Materialize the canonical CS surface-gradient operator."""
        return self.surface_gradient_operator(grid, persist=persist).to_array()

    def surface_gradient_operator(self, grid, *, persist=True):
        """Evaluate the native surface gradient on a target grid."""

        def build():
            native = self.mesh.operators.surface_gradient_operator()
            if self._is_native_grid(grid):
                return native
            return self.remapper.tangential_operator(self.native_grid, grid) @ native

        return self._cached_surface_operator("surface_gradient", grid, build)

    def rhat_cross_gradient_array(self, grid, *, persist=True):
        """Materialize the canonical CS rhat-cross-gradient operator."""
        return self.rhat_cross_gradient_operator(grid, persist=persist).to_array()

    def rhat_cross_gradient_operator(self, grid, *, persist=True):
        """Evaluate the native rhat cross gradient on a target grid."""

        def build():
            native = self.mesh.operators.rhat_cross_gradient_operator()
            if self._is_native_grid(grid):
                return native
            return self.remapper.tangential_operator(self.native_grid, grid) @ native

        return self._cached_surface_operator("rhat_cross_gradient", grid, build)

    def helmholtz_synthesis_array(self, grid, *, persist=True):
        """Materialize the canonical CS Helmholtz synthesis operator."""
        return self.helmholtz_synthesis_operator(grid, persist=persist).to_array()

    def helmholtz_synthesis_operator(self, grid, *, persist=True):
        """Evaluate the native helmholtz synthesis on a target grid."""

        def build():
            native = self.mesh.operators.helmholtz_synthesis_operator()
            if self._is_native_grid(grid):
                return native
            return self.remapper.tangential_operator(self.native_grid, grid) @ native

        return self._cached_surface_operator("helmholtz_synthesis", grid, build)

    def helmholtz_analysis_operator(self, grid, *, sqrt_weights=None):
        """Return sparse constrained analysis on the native grid."""
        if not self._is_native_grid(grid):
            return None
        return self.mesh.operators.helmholtz_analysis_operator(sqrt_weights=sqrt_weights)

    def surface_laplacian_operator(self, r=1.0):
        """Apply the mesh's collocated Laplacian to nodal coefficients."""
        return self.mesh.operators.surface_laplacian_operator(r)

    def mean_free_surface_poisson_operator(self, r=1.0):
        """Invert the native Laplacian with zero area mean."""
        return self.mesh.operators.mean_free_surface_poisson_operator(r)

    def scalar_smoothness_operator(self):
        """Return the mesh's area-weighted scalar-gradient penalty."""
        return self.mesh.operators.scalar_smoothness_operator()

    def helmholtz_smoothness_operator(self):
        """Return the mesh's area-weighted div-curl penalty."""
        return self.mesh.operators.helmholtz_smoothness_operator()


__all__ = ["GlobalCSBasis"]
