"""Spherical basis interface utilities."""

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

from kompe.math import (
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
    take_linear_map,
    vstack_linear_maps,
)
from kompe.math.backend import _is_jax_tracer, get_array_module, readonly_numpy_array
from kompe.math.linear_map import _weighted_cross_product


def _helmholtz_normal_matrix(theta_matrix, phi_matrix, row_scale):
    """Return a normal matrix without a Helmholtz synthesis tensor."""
    xp = get_array_module(theta_matrix, phi_matrix, row_scale)
    theta = xp.asarray(theta_matrix)
    phi = xp.asarray(phi_matrix)
    if theta.shape != phi.shape or theta.ndim != 2:
        raise ValueError("Helmholtz derivative matrices must be matching 2-D arrays.")
    grid_size = theta.shape[0]
    weights = (
        xp.ones((2, grid_size))
        if row_scale is None
        else xp.abs(xp.asarray(row_scale).reshape(2, grid_size)) ** 2
    )
    theta_weights, phi_weights = weights

    # Equal component weights share three products instead of six.
    # This setup-only equality check transfers one boolean under JAX,
    # never the weights or field arrays. Traced weights use the full
    # expression without a host synchronization or data-dependent Python.
    shared_weights = row_scale is None or (
        not _is_jax_tracer(weights) and bool(xp.all(theta_weights == phi_weights))
    )
    if shared_weights:
        diagonal = _weighted_cross_product(theta, theta, theta_weights)
        diagonal += _weighted_cross_product(phi, phi, theta_weights)
        cross = _weighted_cross_product(theta, phi, theta_weights)
        cross -= cross.T.conj()
        first_diagonal = second_diagonal = diagonal
    else:
        first_diagonal = _weighted_cross_product(theta, theta, theta_weights)
        first_diagonal += _weighted_cross_product(phi, phi, phi_weights)
        second_diagonal = _weighted_cross_product(phi, phi, theta_weights)
        second_diagonal += _weighted_cross_product(theta, theta, phi_weights)
        cross = _weighted_cross_product(theta, phi, theta_weights)
        cross -= _weighted_cross_product(phi, theta, phi_weights)
    return xp.block([[first_diagonal, cross], [cross.T.conj(), second_diagonal]])


class ScalarBasis(ABC):
    """Basis capable of synthesizing scalar values on a spherical grid.

    This deliberately does not imply a closed surface, a coefficient-space
    Laplacian, or Helmholtz gauge semantics. Green-function bases such as
    SECS can implement scalar synthesis without making those stronger claims.
    """

    required_attributes = ("kind", "index_names", "coefficient_count", "index_arrays")

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

    def scalar_evaluation_array(self, grid, gradient_component=None, *, persist=True):
        """Materialize scalar values or one unit-sphere gradient component."""
        return self.scalar_evaluation_operator(
            grid, gradient_component=gradient_component, persist=persist
        ).to_array()

    @abstractmethod
    def scalar_evaluation_operator(self, grid, gradient_component=None, *, persist=True):
        """Map coefficients to scalar values or a gradient component.

        ``gradient_component='theta'`` evaluates ``d/dtheta``;
        ``gradient_component='phi'`` evaluates ``(1/sin(theta)) d/dphi``.
        Angles in these differential expressions are radians. Divide
        by the physical radius to obtain a spatial gradient.

        ``persist=False`` forbids disk-cache reads and writes for this
        evaluation. In-memory reuse is independent of this policy.
        A scalar-only implementation may raise NotImplementedError for
        gradient components it does not provide.
        """

    def surface_gradient_array(self, grid, *, persist=True):
        """Materialize the unit-sphere gradient in (theta, phi) order."""
        return self.surface_gradient_operator(grid, persist=persist).to_array()

    def surface_gradient_operator(self, grid, *, persist=True):
        """Return the scalar-to-vector surface-gradient operator."""
        theta = self.scalar_evaluation_operator(grid, gradient_component="theta", persist=persist)
        phi = self.scalar_evaluation_operator(grid, gradient_component="phi", persist=persist)
        return as_linear_map(
            vstack_linear_maps([theta, phi]), output_shape=(2,) + theta.output_shape
        )

    def rhat_cross_gradient_array(self, grid, *, persist=True):
        """Materialize the tangential rhat x grad operator."""
        return self.rhat_cross_gradient_operator(grid, persist=persist).to_array()

    def rhat_cross_gradient_operator(self, grid, *, persist=True):
        """Rotate evaluated components, without storing a rotated gradient."""
        gradient = self.surface_gradient_operator(grid, persist=persist)

        def rotate(values):
            xp = get_array_module(values)
            return xp.stack([-values[1], values[0]])

        def dense_array(xp):
            theta, phi = gradient._dense_array(xp).reshape(2, grid.size, self.coefficient_count)
            return xp.concatenate([-phi, theta])

        return LinearMap(
            shape=gradient.shape,
            dtype=gradient.dtype,
            matvec=lambda c: rotate(gradient.matvec(c).reshape(2, grid.size)).reshape(-1),
            rmatvec=lambda f: gradient.rmatvec(
                -rotate(get_array_module(f).asarray(f).reshape(2, grid.size)).reshape(-1)
            ),
            matmat=lambda c: rotate(gradient.matmat(c).reshape(2, grid.size, -1)).reshape(
                2 * grid.size, -1
            ),
            rmatmat=lambda f: gradient.rmatmat(
                -rotate(f.reshape(2, grid.size, -1)).reshape(2 * grid.size, -1)
            ),
            dense_array=dense_array,
            normal_matrix=lambda xp, row_scale: gradient._normal_matrix(
                xp,
                None
                if row_scale is None
                else xp.asarray(row_scale).reshape(2, grid.size)[::-1].reshape(-1),
            ),
            normal_matrix_diag=lambda row_scale=None: gradient.normal_matrix_diag(
                row_scale=None
                if row_scale is None
                else np.asarray(row_scale).reshape(2, grid.size)[::-1].reshape(-1)
            ),
            backend_operands=gradient.backend_operands,
            input_shape=gradient.input_shape,
            output_shape=gradient.output_shape,
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
    def surface_laplacian_operator(self, r=1.0):
        """Return a Laplacian represented in this same coefficient space.

        This operation requires a space closed under the chosen Laplacian.
        Evaluating derivatives at points does not have that requirement.
        """

    def laplacian_evaluation_operator(self, grid, r=1.0):
        """Map coefficients to surface-Laplacian values at sample points."""
        return self.scalar_evaluation_operator(grid) @ self.surface_laplacian_operator(r)

    def omits_constant_mode(self):
        """Return whether the space cannot represent a nonzero constant.

        This does not imply that all its fields have zero surface mean;
        ``mean_free`` describes that separate property.
        """
        return self.scalar_constant_coefficients is None

    @cached_property
    def helmholtz_gauge_constraints(self):
        """Rows C imposing zero mean on two unobservable constant potentials.

        Return None if constants are absent. A nonzero mean is then observable
        and must not be constrained away. Columns use flat (CF, DF) coefficients.
        """
        if self.omits_constant_mode():
            return None
        return readonly_numpy_array(
            (np.eye(2)[:, :, None] * self.scalar_mean_weights).reshape(2, -1)
        )

    @cached_property
    def mean_free(self):
        """Whether every represented scalar field has zero surface mean."""
        return not np.any(self.scalar_mean_weights)

    @property
    def scalar_constant_coefficients(self):
        """Coefficients of the unit constant field, or None if it is absent.

        This is a field representation, not the surface-mean functional.
        A coefficient subset contains the constant only if it retains
        every coefficient needed to represent that field.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define constant-field coefficients."
        )

    def scalar_smoothness_operator(self):
        """Return R with ||R c||² equal to mean(|grad_s f|²), on a unit sphere.

        The residual may be modal or sampled; it need not be a coefficient
        vector or a physical gradient. Preserve its natural operator structure.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define scalar surface smoothness."
        )

    def helmholtz_smoothness_operator(self):
        """Return R penalizing mean(div_s(F)² + curl_r(F)²) on a unit sphere.

        Coefficients have shape (2, coefficient_count), ordered as curl-free
        and divergence-free potentials. Discrete bases approximate this energy.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define Helmholtz surface smoothness."
        )

    @property
    def scalar_mean_weights(self):
        """Return coefficient weights for the physical surface mean."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define scalar surface-mean weights."
        )

    def scalar_mean(self, coeffs):
        """Return means of (coefficient_count, *batch) scalar coefficients."""
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        if not values.ndim or values.shape[0] != self.coefficient_count:
            raise ValueError("Scalar coefficients must start with coefficient_count.")
        return xp.tensordot(xp.asarray(self.scalar_mean_weights), values, axes=([0], [0]))

    def project_scalar_mean_free(self, coeffs):
        """Subtract the constant field carrying each coefficient array's mean.

        A space lacking a constant may still contain nonzero-mean fields.
        Their means cannot be removed by a constant shift within that space;
        impose a mean constraint during fitting instead.
        """
        if self.mean_free:
            return coeffs
        constant = self.scalar_constant_coefficients
        if constant is None:
            raise ValueError(
                "This basis cannot represent a constant shift; "
                "impose a zero-mean constraint during fitting instead."
            )
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        constant = xp.asarray(constant).reshape(
            (self.coefficient_count,) + (1,) * (values.ndim - 1)
        )
        return values - constant * self.scalar_mean(values)

    def project_helmholtz_mean_free(self, coeffs):
        """Remove means from (2, coefficient_count, *batch) potentials.

        A flat leading axis of length 2*coefficient_count is also accepted
        and retained. Numerical batch axes always trail the field axes.
        """
        if self.mean_free:
            return coeffs
        xp = get_array_module(coeffs)
        values = xp.asarray(coeffs)
        shape = values.shape
        if shape[:2] == (2, self.coefficient_count):
            potentials = values
        elif values.ndim and shape[0] == 2 * self.coefficient_count:
            potentials = values.reshape((2, self.coefficient_count) + shape[1:])
        else:
            raise ValueError("Helmholtz coefficients must start with (2, coefficient_count).")
        projected = self.project_scalar_mean_free(xp.moveaxis(potentials, 1, 0))
        return xp.moveaxis(projected, 0, 1).reshape(shape)

    def helmholtz_synthesis_array(self, grid, *, persist=True):
        """Return the canonical tangential Helmholtz synthesis array.

        Coefficients are ordered as curl-free then divergence-free
        potentials. Components are ordered as theta then phi. The field
        convention is ``-grad(phi) + rhat x grad(psi)``.
        Array axes are ``(component, *sample_axes, potential, coefficient)``.
        """
        return self.helmholtz_synthesis_operator(grid, persist=persist).to_array()

    def helmholtz_synthesis_operator(self, grid, *, persist=True):
        """Evaluate both potential gradients together, then combine components."""
        gradient = self.surface_gradient_operator(grid, persist=persist)
        n, m = self.coefficient_count, grid.size

        def matmat(coefficients):
            xp = get_array_module(coefficients, *gradient.backend_operands)
            columns = coefficients.shape[1]
            potentials = xp.moveaxis(coefficients.reshape(2, n, columns), 0, 1)
            # Axes: derivative component, sample, potential (CF/DF), batch.
            derivatives = gradient.matmat(potentials.reshape(n, 2 * columns)).reshape(
                2, m, 2, columns
            )
            return xp.stack(
                [
                    -derivatives[0, :, 0] - derivatives[1, :, 1],
                    -derivatives[1, :, 0] + derivatives[0, :, 1],
                ]
            ).reshape(2 * m, columns)

        def rmatmat(values):
            xp = get_array_module(values, *gradient.backend_operands)
            columns = values.shape[1]
            theta, phi = values.reshape(2, m, columns)
            components = xp.stack(
                [xp.stack([-theta, phi], axis=1), xp.stack([-phi, -theta], axis=1)]
            )
            potentials = gradient.rmatmat(components.reshape(2 * m, 2 * columns)).reshape(
                n, 2, columns
            )
            return xp.moveaxis(potentials, 1, 0).reshape(2 * n, columns)

        def dense_array(xp):
            theta, phi = gradient._dense_array(xp).reshape(2, m, n)
            return xp.block([[-theta, -phi], [-phi, theta]])

        def normal_matrix_diag(row_scale=None):
            scalar = gradient.normal_matrix_diag(row_scale=row_scale)
            rotated = (
                scalar
                if row_scale is None
                else gradient.normal_matrix_diag(
                    row_scale=np.asarray(row_scale).reshape(2, m)[::-1].reshape(-1)
                )
            )
            return np.concatenate([scalar, rotated])

        def normal_matrix(xp, row_scale):
            theta = self.scalar_evaluation_operator(
                grid, gradient_component="theta", persist=persist
            )._dense_array(xp)
            phi = self.scalar_evaluation_operator(
                grid, gradient_component="phi", persist=persist
            )._dense_array(xp)
            return _helmholtz_normal_matrix(theta, phi, row_scale)

        # Explicit inspection builds the blocks directly, not by probing.
        return LinearMap(
            shape=(2 * m, 2 * n),
            dtype=gradient.dtype,
            matvec=lambda c: matmat(get_array_module(c).asarray(c).reshape(2 * n, 1)).reshape(-1),
            rmatvec=lambda f: rmatmat(get_array_module(f).asarray(f).reshape(2 * m, 1)).reshape(
                -1
            ),
            matmat=matmat,
            rmatmat=rmatmat,
            dense_array=dense_array,
            normal_matrix_diag=normal_matrix_diag,
            normal_matrix=normal_matrix,
            backend_operands=gradient.backend_operands,
            input_shape=(2, n),
            output_shape=gradient.output_shape,
        )

    def helmholtz_curl_free_potential_operator(self):
        """Return the Helmholtz-to-curl-free-potential operator."""
        return take_linear_map((2, self.coefficient_count), 0, axis=0)

    def helmholtz_divergence_free_potential_operator(self):
        """Return the Helmholtz-to-div-free-potential operator."""
        return take_linear_map((2, self.coefficient_count), 1, axis=0)

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
            input_shape=(self.coefficient_count,),
            output_shape=(self.coefficient_count,),
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

        self.kind = parent_basis.kind
        self.index_names = tuple(parent_basis.index_names)
        self.coefficient_count = int(self._parent_coefficient_indices.size)
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
            f"coefficient_count={self.coefficient_count})"
        )

    @staticmethod
    def _normalize_coefficient_indices(parent_basis, coefficient_indices):
        """Return validated parent coefficient indices for a subset."""
        parent_length = int(parent_basis.coefficient_count)
        if coefficient_indices is None:
            return np.arange(parent_length, dtype=int)

        raw_indices = np.asarray(coefficient_indices)
        if raw_indices.ndim != 1:
            raise ValueError("BasisSubset coefficient_indices must be one-dimensional.")
        if raw_indices.dtype == bool:
            if raw_indices.size != parent_length:
                raise ValueError(
                    "BasisSubset boolean coefficient_indices must match parent coefficient_count."
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
            if array.shape == (parent_basis.coefficient_count,):
                arrays.append(readonly_numpy_array(array[coefficient_indices]))
            elif array.size == parent_basis.coefficient_count:
                arrays.append(
                    readonly_numpy_array(
                        array.reshape(parent_basis.coefficient_count)[coefficient_indices]
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
        parent_indices = np.arange(self.parent_basis.coefficient_count, dtype=int)
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

    def _restrict_operator(self, operator):
        """Select coefficient columns without materializing the parent map.

        Reuse an existing array when available. Otherwise retain the parent
        action and its cheap normal diagonal; a subset's normal diagonal is
        exactly the selected parent diagonal, even for rectangular penalties.
        """
        indices = self._parent_coefficient_indices
        input_shape = operator.input_shape[:-1] + (self.coefficient_count,)
        flat_indices = (
            np.arange(operator.shape[1]).reshape(operator.input_shape)[..., indices].reshape(-1)
        )
        matrix = operator.materialized_matrix
        if matrix is not None:
            columns = flat_indices
            if columns.size and np.all(np.diff(columns) == 1):
                columns = slice(int(columns[0]), int(columns[-1]) + 1)
            return as_linear_map(
                matrix[:, columns], input_shape=input_shape, output_shape=operator.output_shape
            )
        selection = take_linear_map(operator.input_shape, indices, axis=-1)
        action = operator @ selection.adjoint()
        return LinearMap(
            shape=action.shape,
            dtype=action.dtype,
            matvec=action.matvec,
            rmatvec=action.rmatvec,
            matmat=action.matmat,
            rmatmat=action.rmatmat,
            dense_array=action._dense_array,
            sparse_matrix=action.to_sparse_matrix if action.is_sparse else None,
            normal_matrix_diag=lambda row_scale=None: operator.normal_matrix_diag(
                row_scale=row_scale
            )[flat_indices],
            backend_operands=action.backend_operands,
            input_shape=input_shape,
            output_shape=operator.output_shape,
        )

    def scalar_evaluation_operator(self, grid, gradient_component=None, *, persist=True):
        """Evaluate the selected coefficients through the parent operator."""
        return self._restrict_operator(
            self.parent_basis.scalar_evaluation_operator(
                grid, gradient_component=gradient_component, persist=persist
            )
        )

    def surface_gradient_operator(self, grid, *, persist=True):
        """Evaluate the complete gradient of the selected basis functions."""
        return self._restrict_operator(
            self.parent_basis.surface_gradient_operator(grid, persist=persist)
        )

    def rhat_cross_gradient_operator(self, grid, *, persist=True):
        """Evaluate the complete rotated gradient of the selected functions."""
        return self._restrict_operator(
            self.parent_basis.rhat_cross_gradient_operator(grid, persist=persist)
        )

    def helmholtz_synthesis_operator(self, grid, *, persist=True):
        """Evaluate both selected potentials through the parent operator."""
        return self._restrict_operator(
            self.parent_basis.helmholtz_synthesis_operator(grid, persist=persist)
        )

    def laplacian_evaluation_operator(self, grid, r=1.0):
        """Evaluate the full derivative without truncating it back to this subset."""
        return self._restrict_operator(self.parent_basis.laplacian_evaluation_operator(grid, r))

    def surface_laplacian_operator(self, r=1.0):
        """Return a same-space Laplacian only when closure is established.

        A diagonal parent Laplacian preserves any subset. A permutation of
        the complete parent space is also closed. Other restrictions require
        an explicit projection, not an implicit truncation of derivatives.
        """
        parent_operator = self.parent_basis.surface_laplacian_operator(r)
        indices = self._parent_coefficient_indices
        if parent_operator.is_diagonal:
            return diagonal_linear_map(
                parent_operator.diagonal()[indices],
                input_shape=(self.coefficient_count,),
                output_shape=(self.coefficient_count,),
            )
        if self.coefficient_count != self.parent_basis.coefficient_count:
            raise NotImplementedError(
                "This subset is not known to be closed under the Laplacian. "
                "Use laplacian_evaluation_operator(grid) for derivative values, "
                "or explicitly compose a projection with the parent Laplacian."
            )
        selection = take_linear_map((self.parent_basis.coefficient_count,), indices)
        return selection @ self._restrict_operator(parent_operator)

    @cached_property
    def scalar_constant_coefficients(self):
        """Restrict the constant only when all of its support is retained."""
        constant = self.parent_basis.scalar_constant_coefficients
        if constant is None:
            return None
        indices = self._parent_coefficient_indices
        if np.count_nonzero(constant[indices]) != np.count_nonzero(constant):
            return None
        return readonly_numpy_array(constant[indices])

    def scalar_smoothness_operator(self):
        """Penalize the complete scalar field represented by this subset."""
        parent = self.parent_basis.scalar_smoothness_operator()
        if parent.is_diagonal:
            return diagonal_linear_map(parent.diagonal()[self._parent_coefficient_indices])
        return self._restrict_operator(parent)

    @property
    def scalar_mean_weights(self):
        """Restrict the parent surface-mean functional to this subset."""
        return self.parent_basis.scalar_mean_weights[self._parent_coefficient_indices]

    def helmholtz_smoothness_operator(self):
        """Penalize both potentials without discarding derivative residuals."""
        parent = self.parent_basis.helmholtz_smoothness_operator()
        if parent.is_diagonal:
            weights = parent.diagonal().reshape(parent.input_shape)[
                :, self._parent_coefficient_indices
            ]
            return diagonal_linear_map(
                weights.reshape(-1),
                input_shape=(2, self.coefficient_count),
                output_shape=(2, self.coefficient_count),
            )
        return self._restrict_operator(parent)

    def with_mean_free(self, mean_free):
        """Return a compatible mean-free/full basis when available."""
        target_mean_free = bool(mean_free)
        if self.mean_free == target_mean_free:
            return self
        if target_mean_free in self._related_basis_cache:
            return self._related_basis_cache[target_mean_free]
        raise NotImplementedError(f"{type(self).__name__} does not define mean-free variants.")
