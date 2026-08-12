"""Spherical basis interface utilities."""

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from kompe.math import LinearMap, as_linear_map
from kompe.math.backend import get_array_module


def _owned_readonly_array(values, *, dtype=None):
    """Return an owned immutable NumPy metadata array."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _backend_stack(values, axis=0):
    """Stack arrays on their active backend."""
    xp = get_array_module(*values)
    return xp.stack([xp.asarray(value) for value in values], axis=axis)


def _coefficient_matrix(value, size, name):
    """Return a square coefficient-space matrix."""
    xp = get_array_module(value)
    array = xp.asarray(value)
    if array.ndim == 1:
        if array.size != size:
            raise ValueError(f"{name} has length {array.size}, expected {size}.")
        return xp.diag(array)
    if array.ndim == 2:
        if array.shape != (size, size):
            raise ValueError(f"{name} has shape {array.shape}, expected {(size, size)}.")
        return array
    raise ValueError(f"{name} must be a 1-D diagonal or 2-D square operator.")


def _helmholtz_component_operator(size, component):
    """Return a structured selector for one Helmholtz potential."""
    component = int(component)
    if component not in (0, 1):
        raise ValueError("Helmholtz component must be 0 or 1.")
    size = int(size)

    def matvec(vec):
        xp = get_array_module(vec)
        values = xp.asarray(vec).reshape(2, size)
        return values[component]

    def rmatvec(vec):
        xp = get_array_module(vec)
        values = xp.asarray(vec).reshape(size)
        zeros = xp.zeros_like(values)
        parts = [values, zeros] if component == 0 else [zeros, values]
        return xp.stack(parts, axis=0).reshape(2 * size)

    def matmat(block):
        xp = get_array_module(block)
        values = xp.asarray(block)
        if values.ndim == 1:
            return matvec(values)
        return values.reshape(2, size, -1)[component].reshape(size, -1)

    def rmatmat(block):
        xp = get_array_module(block)
        values = xp.asarray(block).reshape(size, -1)
        zeros = xp.zeros_like(values)
        parts = [values, zeros] if component == 0 else [zeros, values]
        return xp.stack(parts, axis=0).reshape(2 * size, -1)

    def dense_array(xp):
        identity = xp.eye(size)
        zeros = xp.zeros_like(identity)
        parts = [identity, zeros] if component == 0 else [zeros, identity]
        return xp.concatenate(parts, axis=1)

    def normal_matrix_diag():
        diagonal = np.zeros(2 * size)
        start = component * size
        diagonal[start : start + size] = 1.0
        return diagonal

    return LinearMap(
        shape=(size, 2 * size),
        dtype=np.float64,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        input_shape=(2, size),
        output_shape=(size,),
    )


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
        parts = [type(self).__module__, type(self).__qualname__, self.kind]
        for name in (
            "max_degree",
            "max_order",
            "min_degree",
            "mean_free",
            "backend",
            "quasi_normalized",
            "cells_per_face",
        ):
            if hasattr(self, name):
                parts.append((name, getattr(self, name)))
        return tuple(parts)

    @property
    def coefficient_space_signature(self):
        """Return a signature for coefficient-space compatibility.

        This describes coefficient layout and scaling, not incidental
        implementation choices.
        """
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.kind,
            tuple(self.index_names),
            self.index_length,
        )

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
            raise ValueError(
                f"{type(self).__name__} is missing basis metadata: {joined}."
            )

    @abstractmethod
    def scalar_evaluation_matrix(self, grid, derivative=None):
        """Return the scalar coefficient-to-grid matrix."""

    def _uncached_scalar_evaluation_matrix(self, grid, derivative=None):
        """Evaluate without optional persistent materialization."""
        return self.scalar_evaluation_matrix(grid, derivative=derivative)

    def scalar_evaluation_operator(self, grid, derivative=None):
        """Return the scalar coefficient-to-grid operator."""
        matrix = self.scalar_evaluation_matrix(grid, derivative=derivative)
        return as_linear_map(
            matrix, input_shape=(self.index_length,), output_shape=matrix.shape[:-1]
        )


class SurfaceDifferentialBasis(ScalarBasis):
    """Basis with scalar and vector operators on a spherical surface.

    The shared tangential Helmholtz convention is
    ``F = -grad(phi) + rhat x grad(psi)``, where ``phi`` is the
    curl-free potential and ``psi`` is the divergence-free potential.
    With this convention, ``div_s(F) = -laplacian(phi)`` and the radial
    component of ``curl(F)`` is ``laplacian(psi)``.
    """

    @abstractmethod
    def _surface_laplacian(self, r=1.0):
        """Return the scalar surface Laplacian operator."""

    def surface_gradient_matrix(self, grid):
        """Return ``[d_theta, sin(theta)^-1 d_phi]`` on a surface."""
        return _backend_stack(
            [
                self.scalar_evaluation_matrix(grid, derivative="theta"),
                self.scalar_evaluation_matrix(grid, derivative="phi"),
            ]
        )

    def surface_gradient_operator(self, grid):
        """Return the scalar-to-vector surface-gradient operator."""
        matrix = self.surface_gradient_matrix(grid)
        return as_linear_map(
            matrix, input_shape=(self.index_length,), output_shape=matrix.shape[:-1]
        )

    def rhat_cross_gradient_matrix(self, grid):
        """Return the tangential ``rhat x grad`` operator."""
        grad_theta, grad_phi = self.surface_gradient_matrix(grid)
        return _backend_stack([-grad_phi, grad_theta])

    def rhat_cross_gradient_operator(self, grid):
        """Return the scalar-to-vector ``rhat x grad`` operator."""
        matrix = self.rhat_cross_gradient_matrix(grid)
        return as_linear_map(
            matrix, input_shape=(self.index_length,), output_shape=matrix.shape[:-1]
        )

    def helmholtz_synthesis_matrix(self, grid):
        """Return the canonical tangential Helmholtz synthesis tensor.

        Coefficients are ordered as curl-free then divergence-free
        potentials. Components are ordered as theta then phi. The field
        convention is ``-grad(phi) + rhat x grad(psi)``.
        """
        gradient = self.surface_gradient_matrix(grid)
        rhat_cross_gradient = _backend_stack([-gradient[1], gradient[0]])
        return _backend_stack([-gradient, rhat_cross_gradient], axis=2)

    def helmholtz_synthesis_operator(self, grid):
        """Return the Helmholtz-potential-to-vector operator."""
        curl_free = self.surface_gradient_operator(grid) @ _helmholtz_component_operator(
            self.index_length, 0
        )
        divergence_free = self.rhat_cross_gradient_operator(grid) @ _helmholtz_component_operator(
            self.index_length, 1
        )
        return -curl_free + divergence_free

    def helmholtz_curl_free_potential_matrix(self):
        """Return the Helmholtz-to-curl-free-potential matrix."""
        xp = get_array_module()
        identity = xp.eye(self.index_length)
        return xp.stack([identity, xp.zeros_like(identity)], axis=1)

    def helmholtz_curl_free_potential_operator(self):
        """Return the Helmholtz-to-curl-free-potential operator."""
        return _helmholtz_component_operator(self.index_length, 0)

    def helmholtz_divergence_free_potential_matrix(self):
        """Return the Helmholtz-to-divergence-free-potential matrix."""
        xp = get_array_module()
        identity = xp.eye(self.index_length)
        return xp.stack([xp.zeros_like(identity), identity], axis=1)

    def helmholtz_divergence_free_potential_operator(self):
        """Return the Helmholtz-to-div-free-potential operator."""
        return _helmholtz_component_operator(self.index_length, 1)

    def surface_laplacian_matrix(self, r=1.0):
        """Return the scalar surface-Laplacian coefficient matrix."""
        return _coefficient_matrix(self._surface_laplacian(r), self.index_length, "laplacian")

    def surface_laplacian_operator(self, r=1.0):
        """Return the surface scalar Laplacian operator."""
        return as_linear_map(self._surface_laplacian(r))

    def mean_free_surface_poisson_operator(self, r=1.0):
        """Return the gauge-fixed inverse surface Laplacian.

        Scalar spherical-harmonic spaces represent the surface
        Laplacian diagonally. Mean-free spaces therefore have an exact,
        nonsingular coefficient-space inverse. Nodal bases with a
        constant nullspace should override this method with their
        natural gauge constraint.
        """
        laplacian = self._surface_laplacian(r)
        xp = get_array_module(laplacian)
        values = xp.asarray(laplacian)
        if values.ndim != 1:
            raise NotImplementedError(
                f"{type(self).__name__} must define a gauge-fixed surface Poisson operator."
            )
        if bool(xp.any(values == 0)):
            raise ValueError(
                "The surface Poisson operator requires a mean-free coefficient space."
            )
        return as_linear_map(1.0 / values)

    def helmholtz_surface_divergence_matrix(self, r=1.0):
        """Return the Helmholtz-to-surface-divergence matrix.

        Helmholtz coefficients are ordered as curl-free then
        divergence-free potentials. With the synthesis convention
        ``-grad(phi) + rhat x grad(psi)``, surface divergence is
        ``-laplacian(phi)``.
        """
        laplacian = self.surface_laplacian_matrix(r)
        xp = get_array_module(laplacian)
        return xp.stack([-laplacian, xp.zeros_like(laplacian)], axis=1)

    def helmholtz_surface_divergence_operator(self, r=1.0):
        """Return the Helmholtz-to-surface-divergence operator."""
        return -self.surface_laplacian_operator(r) @ _helmholtz_component_operator(
            self.index_length, 0
        )

    def helmholtz_radial_curl_matrix(self, r=1.0):
        """Return the Helmholtz-coefficient to radial-curl matrix.

        Helmholtz coefficients are ordered as curl-free then
        divergence-free potentials. With the synthesis convention
        ``-grad(phi) + rhat x grad(psi)``, radial curl is
        ``laplacian(psi)``.
        """
        laplacian = self.surface_laplacian_matrix(r)
        xp = get_array_module(laplacian)
        return xp.stack([xp.zeros_like(laplacian), laplacian], axis=1)

    def helmholtz_radial_curl_operator(self, r=1.0):
        """Return the Helmholtz-coefficient to radial-curl operator."""
        return self.surface_laplacian_operator(r) @ _helmholtz_component_operator(
            self.index_length, 1
        )


class BasisView(SurfaceDifferentialBasis):
    """Coefficient-space view of another evaluable basis."""

    def __init__(
        self,
        parent_basis,
        coefficient_indices=None,
        *,
        metadata=None,
        coefficient_space_signature=None,
        view_name="view",
    ):
        """Initialize a coefficient-space view."""
        if not isinstance(parent_basis, SurfaceDifferentialBasis):
            raise TypeError("BasisView parent_basis must implement SurfaceDifferentialBasis.")

        parent_basis.validate_metadata()
        self.parent_basis = parent_basis
        self._parent_coefficient_indices = _owned_readonly_array(
            self._normalize_coefficient_indices(parent_basis, coefficient_indices), dtype=int
        )
        self._view_name = str(view_name)
        self._coefficient_space_signature = coefficient_space_signature
        self._related_basis_cache = {}

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

    @staticmethod
    def _normalize_coefficient_indices(parent_basis, coefficient_indices):
        """Return validated parent coefficient indices for a view."""
        parent_length = int(parent_basis.index_length)
        if coefficient_indices is None:
            return np.arange(parent_length, dtype=int)

        raw_indices = np.asarray(coefficient_indices)
        if raw_indices.ndim != 1:
            raise ValueError("BasisView coefficient_indices must be one-dimensional.")
        if raw_indices.dtype == bool:
            if raw_indices.size != parent_length:
                raise ValueError(
                    "BasisView boolean coefficient_indices must match parent index_length."
                )
            indices = np.flatnonzero(raw_indices)
        else:
            if not np.issubdtype(raw_indices.dtype, np.integer):
                raise TypeError(
                    "BasisView coefficient_indices must be integers or a boolean mask."
                )
            indices = raw_indices.astype(int, copy=False)

        if np.any(indices < 0) or np.any(indices >= parent_length):
            raise IndexError("BasisView coefficient_indices are outside the parent basis.")
        if np.unique(indices).size != indices.size:
            raise ValueError("BasisView coefficient_indices must not contain duplicates.")
        return indices.copy()

    @staticmethod
    def _slice_index_arrays(parent_basis, coefficient_indices):
        """Slice per-coefficient metadata arrays from the parent."""
        arrays = []
        for values in parent_basis.index_arrays:
            array = np.asarray(values)
            if array.shape == (parent_basis.index_length,):
                arrays.append(_owned_readonly_array(array[coefficient_indices]))
            elif array.size == parent_basis.index_length:
                arrays.append(
                    _owned_readonly_array(
                        array.reshape(parent_basis.index_length)[coefficient_indices]
                    )
                )
            else:
                raise ValueError(
                    "BasisView can only slice index_arrays with one value per coefficient."
                )
        return arrays

    @property
    def signature(self):
        """Return a stable cache signature for this basis view."""
        return self.parent_basis.signature + (
            "view",
            self._view_name,
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
            "VIEW",
            self.parent_basis.coefficient_space_signature,
            tuple(int(index) for index in self._parent_coefficient_indices),
        )

    @property
    def root_basis(self):
        """Return the non-view ancestor for this basis view."""
        basis = self.parent_basis
        while isinstance(basis, BasisView):
            basis = basis.parent_basis
        return basis

    def _slice_coefficient_operator(self, values, operator_name):
        """Slice a parent coefficient-space operator to this view."""
        indices = self._parent_coefficient_indices
        if sp.issparse(values):
            expected_shape = (self.parent_basis.index_length, self.parent_basis.index_length)
            if values.shape != expected_shape:
                raise ValueError(
                    f"{operator_name} has shape {values.shape}, expected {expected_shape}."
                )
            return values.tocsr()[indices, :][:, indices]

        xp = get_array_module(values)
        array = xp.asarray(values)
        if array.ndim == 1:
            if array.size != self.parent_basis.index_length:
                raise ValueError(
                    f"{operator_name} has length {array.size}, expected "
                    f"{self.parent_basis.index_length}."
                )
            return array[indices]
        if array.ndim == 2:
            expected_shape = (self.parent_basis.index_length, self.parent_basis.index_length)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{operator_name} has shape {array.shape}, expected {expected_shape}."
                )
            return array[indices][:, indices]
        raise ValueError(f"{operator_name} must be a 1-D or square 2-D coefficient operator.")

    def _slice_evaluation(self, result):
        """Slice parent evaluation columns into this view."""
        indices = self._parent_coefficient_indices
        if indices.size and np.all(np.diff(indices) == 1):
            return result[:, slice(int(indices[0]), int(indices[-1]) + 1)]
        return result[:, indices]

    def scalar_evaluation_matrix(self, grid, derivative=None):
        """Evaluate the viewed basis functions on ``grid``."""
        return self._slice_evaluation(
            self.parent_basis.scalar_evaluation_matrix(grid, derivative=derivative)
        )

    def _uncached_scalar_evaluation_matrix(self, grid, derivative=None):
        """Evaluate the view without persistent materialization."""
        return self._slice_evaluation(
            self.parent_basis._uncached_scalar_evaluation_matrix(grid, derivative=derivative)
        )

    def _surface_laplacian(self, r=1.0):
        """Return the viewed scalar surface Laplacian operator."""
        return self._slice_coefficient_operator(
            self.parent_basis._surface_laplacian(r), "laplacian"
        )

    def scalar_fields_are_mean_free_by_construction(self):
        """Return whether scalar coefficients omit the mean term."""
        return bool(getattr(self, "mean_free", False))

    def scalar_index_length(self, mean_free=None):
        """Return scalar coefficient count."""
        return int(self.scalar_degrees(mean_free=mean_free).size)

    def scalar_degrees(self, mean_free=None):
        """Return harmonic degrees for the requested scalar space."""
        basis = self.with_mean_free(self.mean_free if mean_free is None else bool(mean_free))
        return basis.n

    def scalar_orders(self, mean_free=None):
        """Return harmonic orders for the requested scalar space."""
        basis = self.with_mean_free(self.mean_free if mean_free is None else bool(mean_free))
        return basis.m

    def scalar_index_arrays(self, mean_free=None):
        """Return scalar index arrays for the requested scalar space."""
        basis = self.with_mean_free(self.mean_free if mean_free is None else bool(mean_free))
        return basis.n, basis.m

    def with_mean_free(self, mean_free):
        """Return a compatible mean-free/full basis when available."""
        target_mean_free = bool(mean_free)
        if bool(getattr(self, "mean_free", False)) == target_mean_free:
            return self
        if hasattr(self.parent_basis, "with_mean_free"):
            return self.parent_basis.with_mean_free(target_mean_free)
        raise NotImplementedError(f"{type(self).__name__} does not define mean-free variants.")
