"""Spherical basis interface utilities."""

from abc import ABC, abstractmethod

import numpy as np

from kompe.math import as_linear_map, diagonal_linear_map, take_linear_map
from kompe.math.backend import get_array_module, readonly_numpy_array


def _backend_stack(values, axis=0):
    """Stack arrays on their active backend."""
    xp = get_array_module(*values)
    return xp.stack([xp.asarray(value) for value in values], axis=axis)


class ScalarBasis(ABC):
    """Basis capable of synthesizing scalar values on a spherical grid.

    This deliberately does not imply a closed surface, a coefficient-space
    Laplacian, or Helmholtz gauge semantics. Green-function bases such as
    SECS can implement scalar synthesis without making those stronger claims.
    """

    required_attributes = ("kind", "index_names", "index_length", "index_arrays")

    @property
    def signature(self):
        """Return a stable cache signature for this basis."""
        return (type(self).__module__, type(self).__qualname__, self.coefficient_space_signature)

    @property
    @abstractmethod
    def coefficient_space_signature(self):
        """Return a signature for coefficient-space compatibility.

        This describes coefficient layout and scaling, not incidental
        implementation choices.
        """

    @property
    def root_basis(self):
        """Return the underlying basis, before any coefficient subsets."""
        return self

    def coefficients_are_compatible_with(self, other):
        """Return whether coefficient vectors share operators."""
        return (
            isinstance(other, ScalarBasis)
            and self.coefficient_space_signature == other.coefficient_space_signature
        )

    def validate_metadata(self) -> None:
        """Validate initialized basis metadata."""
        missing = [name for name in self.required_attributes if getattr(self, name, None) is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{type(self).__name__} is missing basis metadata: {joined}.")

    @abstractmethod
    def scalar_evaluation_array(self, grid, derivative=None):
        """Return scalar basis values and optional derivatives on a grid."""

    def _uncached_scalar_evaluation_array(self, grid, derivative=None):
        """Evaluate without optional persistent materialization."""
        return self.scalar_evaluation_array(grid, derivative=derivative)

    def scalar_evaluation_operator(self, grid, derivative=None):
        """Return the scalar coefficient-to-grid operator."""
        array = self.scalar_evaluation_array(grid, derivative=derivative)
        return as_linear_map(
            array, input_shape=(self.index_length,), output_shape=array.shape[:-1]
        )


class SurfaceDifferentialBasis(ScalarBasis):
    """Basis with scalar and vector operators on a spherical surface.

    The shared tangential Helmholtz convention is
    ``F = -grad(phi) + rhat x grad(psi)``, where ``phi`` is the
    curl-free potential and ``psi`` is the divergence-free potential.
    With this convention, ``div_s(F) = -laplacian(phi)`` and the radial
    component of ``curl(F)`` is ``laplacian(psi)``.
    """

    sample_analysis_uses_grid_remapping = False

    @abstractmethod
    def surface_laplacian_operator(self, r=1.0):
        """Return the scalar surface-Laplacian operator."""

    def omits_constant_mode(self):
        """Return whether the coefficient space omits the constant mode."""
        return False

    def scalar_smoothness_weights(self):
        """Return coefficient weights for scalar surface smoothness."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define scalar surface smoothness weights."
        )

    def helmholtz_smoothness_weights(self):
        """Return coefficient weights for tangential Helmholtz smoothness."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define Helmholtz surface smoothness weights."
        )

    @property
    def scalar_mean_weights(self):
        """Return coefficient weights for the physical surface mean."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define scalar surface-mean weights."
        )

    def project_scalar_mean_free(self, coeffs):
        """Project scalar coefficients to zero mean when the basis requires it."""
        if not self.omits_constant_mode():
            raise NotImplementedError(
                f"{type(self).__name__} must define its scalar mean-free projection."
            )
        return coeffs

    def project_helmholtz_mean_free(self, coeffs):
        """Project both Helmholtz potentials to zero mean."""
        if not self.omits_constant_mode():
            raise NotImplementedError(
                f"{type(self).__name__} must define its Helmholtz mean-free projection."
            )
        return coeffs

    def surface_gradient_array(self, grid):
        """Return ``[d_theta, sin(theta)^-1 d_phi]`` on a surface."""
        return _backend_stack(
            [
                self.scalar_evaluation_array(grid, derivative="theta"),
                self.scalar_evaluation_array(grid, derivative="phi"),
            ]
        )

    def surface_gradient_operator(self, grid):
        """Return the scalar-to-vector surface-gradient operator."""
        array = self.surface_gradient_array(grid)
        return as_linear_map(
            array, input_shape=(self.index_length,), output_shape=array.shape[:-1]
        )

    def rhat_cross_gradient_array(self, grid):
        """Return the tangential ``rhat x grad`` operator."""
        grad_theta, grad_phi = self.surface_gradient_array(grid)
        return _backend_stack([-grad_phi, grad_theta])

    def rhat_cross_gradient_operator(self, grid):
        """Return the scalar-to-vector ``rhat x grad`` operator."""
        array = self.rhat_cross_gradient_array(grid)
        return as_linear_map(
            array, input_shape=(self.index_length,), output_shape=array.shape[:-1]
        )

    def helmholtz_synthesis_array(self, grid):
        """Return the canonical tangential Helmholtz synthesis array.

        Coefficients are ordered as curl-free then divergence-free
        potentials. Components are ordered as theta then phi. The field
        convention is ``-grad(phi) + rhat x grad(psi)``.
        """
        gradient = self.surface_gradient_array(grid)
        rhat_cross_gradient = _backend_stack([-gradient[1], gradient[0]])
        return _backend_stack([-gradient, rhat_cross_gradient], axis=2)

    def helmholtz_synthesis_operator(self, grid):
        """Return the Helmholtz-potential-to-vector operator."""
        curl_free = (
            self.surface_gradient_operator(grid) @ self.helmholtz_curl_free_potential_operator()
        )
        divergence_free = (
            self.rhat_cross_gradient_operator(grid)
            @ self.helmholtz_divergence_free_potential_operator()
        )
        return -curl_free + divergence_free

    def helmholtz_curl_free_potential_operator(self):
        """Return the Helmholtz-to-curl-free-potential operator."""
        return take_linear_map((2, self.index_length), 0, axis=0)

    def helmholtz_divergence_free_potential_operator(self):
        """Return the Helmholtz-to-div-free-potential operator."""
        return take_linear_map((2, self.index_length), 1, axis=0)

    def mean_free_surface_poisson_operator(self, r=1.0):
        """Return the gauge-fixed inverse surface Laplacian.

        Scalar spherical-harmonic spaces represent the surface
        Laplacian diagonally. Mean-free spaces therefore have an exact,
        nonsingular coefficient-space inverse. Nodal bases with a
        constant nullspace should override this method with their
        natural gauge constraint.
        """
        laplacian = self.surface_laplacian_operator(r)
        if not laplacian.is_diagonal:
            raise NotImplementedError(
                f"{type(self).__name__} must define a gauge-fixed surface Poisson operator."
            )
        values = laplacian.diagonal()
        xp = get_array_module(values)
        if bool(xp.any(values == 0)):
            raise ValueError(
                "The surface Poisson operator requires a mean-free coefficient space."
            )
        return diagonal_linear_map(
            1.0 / values,
            input_shape=(self.index_length,),
            output_shape=(self.index_length,),
        )

    def helmholtz_surface_divergence_operator(self, r=1.0):
        """Return the Helmholtz-to-surface-divergence operator."""
        return -self.surface_laplacian_operator(r) @ self.helmholtz_curl_free_potential_operator()

    def helmholtz_radial_curl_operator(self, r=1.0):
        """Return the Helmholtz-coefficient to radial-curl operator."""
        return (
            self.surface_laplacian_operator(r)
            @ self.helmholtz_divergence_free_potential_operator()
        )


class BasisSubset(SurfaceDifferentialBasis):
    """A coefficient subset of another evaluable basis."""

    def __init__(
        self,
        parent_basis,
        coefficient_indices=None,
        *,
        metadata=None,
        coefficient_space_signature=None,
        subset_name="subset",
    ):
        """Select coefficients and metadata from ``parent_basis``."""
        if not isinstance(parent_basis, SurfaceDifferentialBasis):
            raise TypeError("BasisSubset parent_basis must implement SurfaceDifferentialBasis.")

        parent_basis.validate_metadata()
        self.parent_basis = parent_basis
        self._parent_coefficient_indices = readonly_numpy_array(
            self._normalize_coefficient_indices(parent_basis, coefficient_indices), dtype=int
        )
        self._subset_name = str(subset_name)
        self._coefficient_space_signature = coefficient_space_signature
        self._related_basis_cache = {}
        self.mean_free = parent_basis.omits_constant_mode()

        self.kind = parent_basis.kind
        self.index_names = tuple(parent_basis.index_names)
        self.index_length = int(self._parent_coefficient_indices.size)
        self.index_arrays = tuple(
            self._slice_index_arrays(parent_basis, self._parent_coefficient_indices)
        )
        for name, values in zip(self.index_names, self.index_arrays, strict=True):
            if isinstance(name, str) and name.isidentifier() and not hasattr(self, name):
                setattr(self, name, values)

        for name, value in (metadata or {}).items():
            setattr(self, name, value)

        self.validate_metadata()

    def __repr__(self):
        """Summarize the selected coefficient space."""
        return (
            f"BasisSubset(kind={self.kind!r}, subset_name={self._subset_name!r}, "
            f"index_length={self.index_length})"
        )

    @staticmethod
    def _normalize_coefficient_indices(parent_basis, coefficient_indices):
        """Return validated parent coefficient indices for a subset."""
        parent_length = int(parent_basis.index_length)
        if coefficient_indices is None:
            return np.arange(parent_length, dtype=int)

        raw_indices = np.asarray(coefficient_indices)
        if raw_indices.ndim != 1:
            raise ValueError("BasisSubset coefficient_indices must be one-dimensional.")
        if raw_indices.dtype == bool:
            if raw_indices.size != parent_length:
                raise ValueError(
                    "BasisSubset boolean coefficient_indices must match parent index_length."
                )
            indices = np.flatnonzero(raw_indices)
        else:
            if not np.issubdtype(raw_indices.dtype, np.integer):
                raise TypeError(
                    "BasisSubset coefficient_indices must be integers or a boolean mask."
                )
            indices = raw_indices.astype(int, copy=False)

        if np.any(indices < 0) or np.any(indices >= parent_length):
            raise IndexError("BasisSubset coefficient_indices are outside the parent basis.")
        if np.unique(indices).size != indices.size:
            raise ValueError("BasisSubset coefficient_indices must not contain duplicates.")
        return indices.copy()

    @staticmethod
    def _slice_index_arrays(parent_basis, coefficient_indices):
        """Slice per-coefficient metadata arrays from the parent."""
        arrays = []
        for values in parent_basis.index_arrays:
            array = np.asarray(values)
            if array.shape == (parent_basis.index_length,):
                arrays.append(readonly_numpy_array(array[coefficient_indices]))
            elif array.size == parent_basis.index_length:
                arrays.append(
                    readonly_numpy_array(
                        array.reshape(parent_basis.index_length)[coefficient_indices]
                    )
                )
            else:
                raise ValueError(
                    "BasisSubset can only slice index_arrays with one value per coefficient."
                )
        return arrays

    @property
    def signature(self):
        """Return a stable cache signature for this basis subset."""
        return self.parent_basis.signature + (
            "subset",
            self._subset_name,
            tuple(int(index) for index in self._parent_coefficient_indices),
            self.coefficient_space_signature,
        )

    @property
    def coefficient_space_signature(self):
        """Return a signature for coefficient-space compatibility."""
        if self._coefficient_space_signature is not None:
            return self._coefficient_space_signature
        parent_indices = np.arange(self.parent_basis.index_length, dtype=int)
        if np.array_equal(self._parent_coefficient_indices, parent_indices):
            return self.parent_basis.coefficient_space_signature
        return (
            "SUBSET",
            self.parent_basis.coefficient_space_signature,
            tuple(int(index) for index in self._parent_coefficient_indices),
        )

    @property
    def root_basis(self):
        """Return the first ancestor that is not a subset."""
        return self.parent_basis.root_basis

    def _slice_evaluation(self, result):
        """Slice parent evaluation columns into this subset."""
        indices = self._parent_coefficient_indices
        if indices.size and np.all(np.diff(indices) == 1):
            return result[:, slice(int(indices[0]), int(indices[-1]) + 1)]
        return result[:, indices]

    def scalar_evaluation_array(self, grid, derivative=None):
        """Evaluate the selected basis functions on ``grid``."""
        return self._slice_evaluation(
            self.parent_basis.scalar_evaluation_array(grid, derivative=derivative)
        )

    def _uncached_scalar_evaluation_array(self, grid, derivative=None):
        """Evaluate the subset without persistent materialization."""
        return self._slice_evaluation(
            self.parent_basis._uncached_scalar_evaluation_array(grid, derivative=derivative)
        )

    def surface_laplacian_operator(self, r=1.0):
        """Return the parent surface Laplacian restricted to this subset."""
        parent_operator = self.parent_basis.surface_laplacian_operator(r)
        indices = self._parent_coefficient_indices
        if parent_operator.is_diagonal:
            return diagonal_linear_map(
                parent_operator.diagonal()[indices],
                input_shape=(self.index_length,),
                output_shape=(self.index_length,),
            )
        selection = take_linear_map((self.parent_basis.index_length,), indices)
        return selection @ parent_operator @ selection.adjoint()

    def omits_constant_mode(self):
        """Return whether scalar coefficients omit the mean term."""
        return self.mean_free

    def scalar_smoothness_weights(self):
        """Return the parent basis's scalar smoothness weights on this subset."""
        parent_weights = self.parent_basis.scalar_smoothness_weights()
        return np.asarray(parent_weights)[self._parent_coefficient_indices]

    def helmholtz_smoothness_weights(self):
        """Return the parent basis's Helmholtz smoothness weights on this subset."""
        parent_weights = self.parent_basis.helmholtz_smoothness_weights()
        return np.asarray(parent_weights)[self._parent_coefficient_indices]

    def with_mean_free(self, mean_free):
        """Return a compatible mean-free/full basis when available."""
        target_mean_free = bool(mean_free)
        if self.mean_free == target_mean_free:
            return self
        if hasattr(self.parent_basis, "with_mean_free"):
            return self.parent_basis.with_mean_free(target_mean_free)
        raise NotImplementedError(f"{type(self).__name__} does not define mean-free variants.")
