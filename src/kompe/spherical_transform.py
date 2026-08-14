"""Spherical transform module.

This module contains the SphericalTransform class for converting between
spherical-basis coefficients and grid values.
"""

from collections import OrderedDict
from functools import cached_property

import numpy as np
from scipy.linalg import cholesky

from kompe.core import SurfaceDifferentialBasis
from kompe.grid import SphericalGrid
from kompe.math import array_fingerprint
from kompe.math.backend import get_array_module
from kompe.math.least_squares_problem import LeastSquaresProblem
from kompe.math.least_squares_solver import (
    LeastSquaresSolver,
    factorized_least_squares_map,
    get_default_least_squares_solver,
)
from kompe.math.linear_map import (
    as_linear_map,
    diagonal_linear_map,
    is_noop_linear_map,
)
from kompe.math.tensor_operations import weighted_tensor_pinv

_LEAST_SQUARES_CACHE_VERSION = 2
_WEIGHTED_PRODUCT_WORK_BYTES = 512 * 1024**2


def grid_sqrt_area_weights(grid):
    """Return default sqrt area weights for a spherical grid."""
    if hasattr(grid, "area_weights"):
        xp = get_array_module(grid.area_weights)
        weights = xp.asarray(grid.area_weights, dtype=float)
    else:
        xp = get_array_module(grid.theta)
        theta = xp.asarray(grid.theta, dtype=float)
        weights = xp.sin(xp.deg2rad(theta))
    weights = xp.clip(weights, 0.0, None)
    return xp.sqrt(weights)


def resolve_sqrt_weights(grid, sqrt_weights=None, area_weighted=False, vector=False):
    """Resolve explicit or default grid sqrt weights."""
    if sqrt_weights is not None:
        return sqrt_weights
    if not area_weighted:
        return None
    weights = grid_sqrt_area_weights(grid)
    xp = get_array_module(weights)
    return xp.tile(weights, (2, 1)) if vector else weights


def _scalar_surface_smoothness_weights(degree):
    """Return scalar surface-gradient-energy square-root weights.

    The real Schmidt basis used here has angular norm
    ``q_n = 1 / (2 n + 1)``.  The squared surface-gradient norm of a
    degree-``n`` scalar coefficient is therefore
    ``q_n n (n + 1)``.
    """
    degree = np.asarray(degree, dtype=float)
    angular_norm = 1.0 / (2.0 * degree + 1.0)
    laplacian_eigenvalue = degree * (degree + 1.0)
    return np.sqrt(angular_norm * laplacian_eigenvalue)


def _helmholtz_surface_smoothness_weights(degree):
    """Return equal curl-free/divergence-free field-smoothness weights.

    For ``F = -grad(phi) + rhat x grad(psi)``, penalizing the surface
    divergence and curl of ``F`` gives the same degree weight for both
    Helmholtz potentials: ``sqrt(q_n) n (n + 1)``.
    """
    degree = np.asarray(degree, dtype=float)
    angular_norm = 1.0 / (2.0 * degree + 1.0)
    laplacian_eigenvalue = degree * (degree + 1.0)
    return np.sqrt(angular_norm) * laplacian_eigenvalue


def _owned_explicit_weights(values):
    """Own weights so cached analysis cannot be mutated externally."""
    owned = np.array(values, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _helmholtz_squared_weights(sqrt_weights, grid_size):
    """Return objective weights by vector component and point."""
    if sqrt_weights is None:
        return np.ones((2, grid_size))
    values = np.asarray(sqrt_weights, dtype=float)
    if values.size != 2 * grid_size:
        raise ValueError(
            f"Helmholtz sqrt_weights must contain {2 * grid_size} values; got {values.size}."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Helmholtz sqrt_weights must be finite and non-negative.")
    return values.reshape(2, grid_size) ** 2


def _weighted_cross_product(left, right, weights):
    """Return ``left* diag(weights) right`` in bounded row blocks."""
    left = np.asarray(left)
    right = np.asarray(right)
    weights = np.asarray(weights)
    if left.shape[0] != right.shape[0] or weights.shape != (left.shape[0],):
        raise ValueError("Weighted cross-product operands have incompatible shapes.")
    dtype = np.result_type(left.dtype, right.dtype, weights.dtype)
    result = np.zeros((left.shape[1], right.shape[1]), dtype=dtype)
    bytes_per_row = max(1, right.shape[1] * np.dtype(dtype).itemsize)
    rows_per_block = max(1, _WEIGHTED_PRODUCT_WORK_BYTES // bytes_per_row)
    for start in range(0, left.shape[0], rows_per_block):
        stop = min(left.shape[0], start + rows_per_block)
        weighted_right = weights[start:stop, None] * right[start:stop]
        result += left[start:stop].T.conj() @ weighted_right
    return result


def _scalar_data_normal_matrix(matrix, sqrt_weights):
    """Return a weighted scalar data-term normal matrix."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Scalar synthesis must be a 2-D matrix.")
    if sqrt_weights is None:
        weights = np.ones(matrix.shape[0])
    else:
        weights = np.asarray(sqrt_weights, dtype=float).reshape(-1)
        if weights.shape != (matrix.shape[0],):
            raise ValueError("Scalar sqrt_weights must match the grid size.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Scalar sqrt_weights must be finite and non-negative.")
        weights = weights**2
    return _weighted_cross_product(matrix, matrix, weights)


def _helmholtz_data_normal_matrix(theta_matrix, phi_matrix, sqrt_weights):
    """Return a normal matrix without a Helmholtz synthesis tensor."""
    theta = np.asarray(theta_matrix)
    phi = np.asarray(phi_matrix)
    if theta.shape != phi.shape or theta.ndim != 2:
        raise ValueError("Helmholtz derivative matrices must be matching 2-D arrays.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("Helmholtz derivative matrices must contain only finite values.")

    grid_size, coefficient_size = theta.shape
    theta_weights, phi_weights = _helmholtz_squared_weights(sqrt_weights, grid_size)
    normal = np.empty((2 * coefficient_size, 2 * coefficient_size), order="F")

    if np.array_equal(theta_weights, phi_weights):
        diagonal = _weighted_cross_product(theta, theta, theta_weights)
        diagonal += _weighted_cross_product(phi, phi, theta_weights)
        cross = _weighted_cross_product(theta, phi, theta_weights)
        cross -= cross.T.conj()
        normal[:coefficient_size, :coefficient_size] = diagonal
        normal[coefficient_size:, coefficient_size:] = diagonal
    else:
        first_diagonal = _weighted_cross_product(theta, theta, theta_weights)
        first_diagonal += _weighted_cross_product(phi, phi, phi_weights)
        second_diagonal = _weighted_cross_product(phi, phi, theta_weights)
        second_diagonal += _weighted_cross_product(theta, theta, phi_weights)
        cross = _weighted_cross_product(theta, phi, theta_weights)
        cross -= _weighted_cross_product(phi, theta, phi_weights)
        normal[:coefficient_size, :coefficient_size] = first_diagonal
        normal[coefficient_size:, coefficient_size:] = second_diagonal

    normal[:coefficient_size, coefficient_size:] = cross
    normal[coefficient_size:, :coefficient_size] = cross.T.conj()
    return normal


def _helmholtz_normal_factor(theta_matrix, phi_matrix, sqrt_weights):
    """Build a Cholesky factor for Helmholtz analysis."""
    normal = _helmholtz_data_normal_matrix(theta_matrix, phi_matrix, sqrt_weights)
    try:
        return cholesky(normal, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Helmholtz synthesis must have full column rank.") from exc


def _rhat_cross_gradient_normal_factor(theta_matrix, phi_matrix, sqrt_weights, coefficient_scale):
    """Build a normal factor for one rotated-gradient potential."""
    theta = np.asarray(theta_matrix)
    phi = np.asarray(phi_matrix)
    scale = np.asarray(coefficient_scale)
    if theta.shape != phi.shape or theta.ndim != 2:
        raise ValueError("Derivative matrices must be matching 2-D arrays.")
    if scale.shape != (theta.shape[1],):
        raise ValueError("coefficient_scale must match the basis length.")
    theta_weights, phi_weights = _helmholtz_squared_weights(sqrt_weights, theta.shape[0])
    normal = _weighted_cross_product(phi, phi, theta_weights)
    normal += _weighted_cross_product(theta, theta, phi_weights)
    normal *= scale.conj()[:, None] * scale[None, :]
    try:
        return cholesky(normal, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Rotated-gradient synthesis must have full column rank.") from exc


class SphericalTransform:
    """Two-way transform between a spherical basis and a grid.

    This class owns both synthesis (coefficients to grid values) and
    analysis (grid values to coefficients) for scalar and tangential
    Helmholtz fields. It also handles batched analysis of samples from
    external input grids when a grid-remap basis is supplied.
    """

    _input_transform_cache_size = 16
    _cached_attribute_names = (
        "scalar_synthesis_matrix",
        "scalar_synthesis_operator",
        "theta_derivative_matrix",
        "theta_derivative_operator",
        "phi_derivative_matrix",
        "phi_derivative_operator",
        "surface_gradient_matrix",
        "surface_gradient_operator",
        "rhat_cross_gradient_matrix",
        "rhat_cross_gradient_operator",
        "helmholtz_synthesis_matrix",
        "helmholtz_synthesis_operator",
        "_optimized_helmholtz_analysis_operator",
        "helmholtz_analysis_operator",
        "scalar_regularization_operator",
        "helmholtz_regularization_operator",
        "scalar_least_squares_problem",
        "helmholtz_least_squares_problem",
    )

    def __init__(
        self,
        basis,
        grid,
        *,
        grid_remap_basis=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        area_weighted=False,
        use_persistent_evaluation_cache=True,
    ):
        """Initialize synthesis and analysis between ``basis`` and ``grid``.

        Parameters
        ----------
        basis : SurfaceDifferentialBasis
            Coefficient representation to evaluate or fit.
        grid : SphericalGrid
            Sample positions in the same spherical coordinate frame as the
            basis.
        grid_remap_basis : SurfaceDifferentialBasis, optional
            Representation used only when external sample grids must first be
            remapped to ``grid``.
        sqrt_weights : array-like, optional
            Square-root residual weights. Their squares weight the
            least-squares objective.
        reg_lambda : float, optional
            Dimensionless relative regularization strength. Kompe balances
            the data and regularization operator scales before solving.
        pinv_rtol : float, optional
            Relative singular-value cutoff used by pseudo-inverse solvers.
        area_weighted : bool, optional
            Use ``grid.area_weights`` when present, otherwise spherical
            ``sin(theta)`` weights. Explicit ``sqrt_weights`` take precedence.
            This affects analysis only, never synthesis.
        use_persistent_evaluation_cache : bool, optional
            Reuse deterministic basis-evaluation matrices through the basis
            cache when available.
        """
        if not isinstance(basis, SurfaceDifferentialBasis):
            raise TypeError("SphericalTransform basis must implement SurfaceDifferentialBasis.")
        if not isinstance(grid, SphericalGrid):
            raise TypeError("SphericalTransform grid must be a SphericalGrid.")
        basis.validate_metadata()
        self.basis = basis
        self.grid = grid
        self.grid_remap_basis = grid_remap_basis
        self.explicit_sqrt_weights = sqrt_weights is not None
        self.area_weighted = bool(area_weighted)
        self.sqrt_weights = resolve_sqrt_weights(
            grid, sqrt_weights=sqrt_weights, area_weighted=area_weighted
        )
        self.helmholtz_sqrt_weights = resolve_sqrt_weights(
            grid, sqrt_weights=sqrt_weights, area_weighted=area_weighted, vector=True
        )
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol
        self.use_persistent_evaluation_cache = bool(use_persistent_evaluation_cache)

        self._input_transforms = OrderedDict()

    def __repr__(self):
        """Summarize the bound basis, grid, and analysis policy."""
        return (
            f"SphericalTransform(basis={self.basis!r}, grid={self.grid!r}, "
            f"area_weighted={self.area_weighted}, reg_lambda={self.reg_lambda!r})"
        )

    def clear_cache(self):
        """Discard matrices, operators, factorizations, and input transforms."""
        for name in self._cached_attribute_names:
            self.__dict__.pop(name, None)
        self._input_transforms.clear()

    def cache_info(self):
        """Return local transform-cache occupancy and configuration."""
        materialized = sum(name in self.__dict__ for name in self._cached_attribute_names)
        return {
            "materialized_values": materialized,
            "scalar_factorization": "scalar_least_squares_problem" in self.__dict__,
            "helmholtz_factorization": "helmholtz_least_squares_problem" in self.__dict__,
            "input_transforms": len(self._input_transforms),
            "input_transform_max_size": self._input_transform_cache_size,
        }

    def _evaluate_basis_on_grid(self, derivative=None):
        """Evaluate the basis on the transform grid."""
        if not self.use_persistent_evaluation_cache:
            return self.basis._uncached_scalar_evaluation_matrix(self.grid, derivative=derivative)
        return self.basis.scalar_evaluation_matrix(self.grid, derivative=derivative)

    def _operator_cache(self):
        """Return the basis's persistent operator cache."""
        basis = getattr(self.basis, "root_basis", self.basis)
        return getattr(basis, "operator_cache", None)

    def _data_normal_matrix_builder(self, field_type):
        """Return a memory-bounded SH normal-matrix builder."""
        from kompe.spherical_harmonics.sh_basis import SHBasis

        root_basis = getattr(self.basis, "root_basis", self.basis)
        if get_array_module() is not np or not isinstance(root_basis, SHBasis):
            return None
        if field_type == "scalar":
            return lambda: _scalar_data_normal_matrix(
                self.scalar_synthesis_matrix, self.sqrt_weights
            )
        if field_type == "helmholtz":
            return lambda: _helmholtz_data_normal_matrix(
                self.theta_derivative_matrix,
                self.phi_derivative_matrix,
                self.helmholtz_sqrt_weights,
            )
        raise ValueError(f"Unknown transform field type {field_type!r}.")

    def _least_squares_cache_identity(self, field_type):
        """Return an exact transform-analysis identity."""
        if self._operator_cache() is None:
            return None
        weights = self.helmholtz_sqrt_weights if field_type == "helmholtz" else self.sqrt_weights
        return {
            "algorithm": "spherical_transform_least_squares",
            "version": _LEAST_SQUARES_CACHE_VERSION,
            "field_type": str(field_type),
            "basis": self.basis.signature,
            "grid_coordinates": self.grid.exact_coordinate_signature,
            "sqrt_weights": array_fingerprint(weights),
            "regularization_lambda": self.reg_lambda,
            "area_weighted": self.area_weighted,
            "normal_matrix_algorithm": (
                "structured_sh_v1"
                if self._data_normal_matrix_builder(field_type) is not None
                else "dense_backend_v1"
            ),
        }

    @cached_property
    def scalar_synthesis_matrix(self):
        """Matrix mapping scalar coefficients to grid values."""
        return self._evaluate_basis_on_grid()

    @cached_property
    def scalar_synthesis_operator(self):
        """Operator mapping scalar coefficients to grid values."""
        if self.use_persistent_evaluation_cache:
            return self.basis.scalar_evaluation_operator(self.grid)
        return as_linear_map(
            self.scalar_synthesis_matrix,
            input_shape=(self.basis.index_length,),
            output_shape=(self.grid.size,),
        )

    @cached_property
    def theta_derivative_matrix(self):
        """Matrix evaluating the theta derivative."""
        gradient = self.__dict__.get("surface_gradient_matrix")
        return gradient[0] if gradient is not None else self._evaluate_basis_on_grid("theta")

    @cached_property
    def theta_derivative_operator(self):
        """Operator evaluating the theta derivative."""
        if self.use_persistent_evaluation_cache:
            return self.basis.scalar_evaluation_operator(self.grid, derivative="theta")
        return as_linear_map(
            self.theta_derivative_matrix,
            input_shape=(self.basis.index_length,),
            output_shape=(self.grid.size,),
        )

    @cached_property
    def phi_derivative_matrix(self):
        """Matrix evaluating the phi derivative."""
        gradient = self.__dict__.get("surface_gradient_matrix")
        return gradient[1] if gradient is not None else self._evaluate_basis_on_grid("phi")

    @cached_property
    def phi_derivative_operator(self):
        """Operator evaluating the phi derivative."""
        if self.use_persistent_evaluation_cache:
            return self.basis.scalar_evaluation_operator(self.grid, derivative="phi")
        return as_linear_map(
            self.phi_derivative_matrix,
            input_shape=(self.basis.index_length,),
            output_shape=(self.grid.size,),
        )

    @cached_property
    def surface_gradient_matrix(self):
        """Matrix evaluating the horizontal gradient."""
        return self.basis.surface_gradient_matrix(self.grid)

    @cached_property
    def surface_gradient_operator(self):
        """Operator evaluating the horizontal gradient."""
        return self.basis.surface_gradient_operator(self.grid)

    @cached_property
    def rhat_cross_gradient_matrix(self):
        """Matrix evaluating r-hat x horizontal gradient."""
        return self.basis.rhat_cross_gradient_matrix(self.grid)

    @cached_property
    def rhat_cross_gradient_operator(self):
        """Operator evaluating r-hat x horizontal gradient."""
        return self.basis.rhat_cross_gradient_operator(self.grid)

    @cached_property
    def helmholtz_synthesis_matrix(self):
        """Matrix evaluating horizontal vector field expansions."""
        gradient = self.__dict__.get("surface_gradient_matrix")
        rotated_gradient = self.__dict__.get("rhat_cross_gradient_matrix")
        if gradient is None and rotated_gradient is None:
            return self.basis.helmholtz_synthesis_matrix(self.grid)
        gradient = self.surface_gradient_matrix
        rotated_gradient = self.rhat_cross_gradient_matrix
        xp = get_array_module(gradient, rotated_gradient)
        return xp.stack([-xp.asarray(gradient), xp.asarray(rotated_gradient)], axis=2)

    @cached_property
    def helmholtz_synthesis_operator(self):
        """Operator evaluating horizontal vector field expansions."""
        return self.basis.helmholtz_synthesis_operator(self.grid)

    @cached_property
    def helmholtz_analysis_operator(self):
        """Map gridded vectors to unregularized coefficients."""
        if self.reg_lambda is not None:
            raise RuntimeError(
                "helmholtz_analysis_operator is only available for unregularized "
                "transforms; use analyze_helmholtz() for a regularized fit."
            )
        optimized = self._optimized_helmholtz_analysis_operator
        if optimized is not None:
            return optimized
        analysis = weighted_tensor_pinv(
            self.helmholtz_synthesis_matrix,
            sqrt_weights=self.helmholtz_sqrt_weights,
            n_leading_flattened=2,
        )
        return as_linear_map(
            analysis,
            input_shape=(2, self.grid.size),
            output_shape=(2, self.basis.index_length),
        )

    @cached_property
    def _optimized_helmholtz_analysis_operator(self):
        """Return an available structured or factorized analysis map."""
        factory = getattr(self.basis, "helmholtz_analysis_operator", None)
        operator = (
            factory(self.grid, sqrt_weights=self.helmholtz_sqrt_weights)
            if callable(factory)
            else None
        )
        if operator is not None:
            return operator
        return self._factorized_helmholtz_analysis_operator()

    def _factorized_helmholtz_analysis_operator(self):
        """Factor analysis when both potentials omit their gauges."""
        if not self.basis.scalar_fields_are_mean_free_by_construction():
            return None

        theta_matrix = self.theta_derivative_matrix
        phi_matrix = self.phi_derivative_matrix
        try:
            cache = self._operator_cache()
            identity = self._least_squares_cache_identity("helmholtz")
            if cache is not None and identity is not None:
                factor = cache.get_or_create(
                    "least_squares_factor",
                    {**identity, "factorization": "structured_helmholtz_cholesky"},
                    lambda: _helmholtz_normal_factor(
                        theta_matrix, phi_matrix, self.helmholtz_sqrt_weights
                    ),
                )
            else:
                factor = _helmholtz_normal_factor(
                    theta_matrix, phi_matrix, self.helmholtz_sqrt_weights
                )
            return factorized_least_squares_map(
                self.helmholtz_synthesis_operator,
                factor,
                sqrt_weights=self.helmholtz_sqrt_weights,
                input_shape=(2, self.grid.size),
                output_shape=(2, self.basis.index_length),
            )
        except ValueError:
            return None

    def rhat_cross_gradient_analysis_operator(self, *, coefficient_scale=None):
        """Factor analysis for one rotated-gradient potential."""
        if coefficient_scale is None:
            coefficient_scale = np.ones(self.basis.index_length)
        scale = np.asarray(coefficient_scale)
        factor = _rhat_cross_gradient_normal_factor(
            self.theta_derivative_matrix,
            self.phi_derivative_matrix,
            self.helmholtz_sqrt_weights,
            scale,
        )
        synthesis = self.rhat_cross_gradient_operator
        if not np.all(scale == 1):
            synthesis = synthesis @ diagonal_linear_map(scale)
        return factorized_least_squares_map(
            synthesis,
            factor,
            sqrt_weights=self.helmholtz_sqrt_weights,
            input_shape=(2, self.grid.size),
            output_shape=(self.basis.index_length,),
        )

    @cached_property
    def scalar_regularization_operator(self):
        """Surface-gradient smoothness operator for scalar fields."""
        if self.reg_lambda is None:
            return None
        if not hasattr(self.basis, "n"):
            raise NotImplementedError("Degree-weighted scalar regularization requires basis.n.")
        return diagonal_linear_map(
            _scalar_surface_smoothness_weights(self.basis.n),
            input_shape=(self.basis.index_length,),
            output_shape=(self.basis.index_length,),
        )

    @cached_property
    def helmholtz_regularization_operator(self):
        """Return equal-component Helmholtz-field smoothness."""
        if self.reg_lambda is None:
            return None
        if not hasattr(self.basis, "n"):
            raise NotImplementedError("Degree-weighted Helmholtz regularization requires basis.n.")
        weights = np.broadcast_to(
            _helmholtz_surface_smoothness_weights(self.basis.n),
            (2, self.basis.index_length),
        )
        return diagonal_linear_map(
            weights.reshape(-1),
            input_shape=(2, self.basis.index_length),
            output_shape=(2, self.basis.index_length),
        )

    @cached_property
    def scalar_least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for scalar fields."""
        return LeastSquaresProblem(
            A=self.scalar_synthesis_operator,
            solution_shape=self.basis.index_length,
            data_shapes=self.grid.size,
            sqrt_weights=self.sqrt_weights,
            regularization_weights=self.reg_lambda,
            regularization_matrices=self.scalar_regularization_operator,
            operator_cache=self._operator_cache(),
            cache_identity=self._least_squares_cache_identity("scalar"),
            data_normal_matrix_builder=self._data_normal_matrix_builder("scalar"),
        )

    @cached_property
    def helmholtz_least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for horizontal vector fields."""
        return LeastSquaresProblem(
            A=self.helmholtz_synthesis_operator,
            solution_shape=(2, self.basis.index_length),
            data_shapes=(2, self.grid.size),
            sqrt_weights=self.helmholtz_sqrt_weights,
            regularization_weights=self.reg_lambda,
            regularization_matrices=self.helmholtz_regularization_operator,
            operator_cache=self._operator_cache(),
            cache_identity=self._least_squares_cache_identity("helmholtz"),
            data_normal_matrix_builder=self._data_normal_matrix_builder("helmholtz"),
        )

    def _solve_least_squares(self, problem, grid_values, solver_type=None):
        """Solve one configured least-squares problem."""
        solver_type = solver_type or get_default_least_squares_solver()
        solver = LeastSquaresSolver(solver=solver_type, tolerance=self.pinv_rtol)
        return solver.solve(problem=problem, rhs=grid_values)

    def synthesize_scalar(self, coeffs, derivative=None):
        """Synthesize scalar coefficients on the transform grid."""
        coeff_array = self._coefficient_array(coeffs, preserve_backend=True)
        return self._coefficients_to_grid(coeff_array, derivative=derivative)

    def synthesize_helmholtz(self, coeffs):
        """Synthesize Helmholtz coefficients on the transform grid."""
        coeff_array = self._coefficient_array(coeffs, helmholtz=True, preserve_backend=True)
        return self._coefficients_to_grid(coeff_array, helmholtz=True)

    def analyze_scalar(self, grid_values, solver_type=None):
        """Analyze scalar grid values into basis coefficients."""
        if self._scalar_analysis_is_noop():
            return grid_values
        return self._solve_least_squares(
            self.scalar_least_squares_problem, grid_values, solver_type
        )

    def analyze_helmholtz(self, grid_values, solver_type=None):
        """Analyze grid values into Helmholtz coefficients."""
        if solver_type is None and self.reg_lambda is None:
            operator = self._optimized_helmholtz_analysis_operator
            if operator is not None:
                return self._apply_helmholtz_analysis_operator(operator, grid_values)
        return self._solve_least_squares(
            self.helmholtz_least_squares_problem, grid_values, solver_type
        )

    def _apply_helmholtz_analysis_operator(self, operator, grid_values):
        """Apply an analysis operator with standard RHS shapes."""
        xp = get_array_module(grid_values)
        values = xp.asarray(grid_values)
        data_shape = (2, self.grid.size)
        output_shape = (2, self.basis.index_length)
        data_size = int(np.prod(data_shape))

        if values.shape == data_shape:
            return operator.matvec(values).reshape(output_shape)
        if values.ndim > 2 and tuple(values.shape[:2]) == data_shape:
            rhs_shape = values.shape[2:]
            value_block = values.reshape(data_size, -1)
        elif values.ndim > 2 and tuple(values.shape[-2:]) == data_shape:
            rhs_shape = values.shape[:-2]
            value_block = values.reshape(-1, data_size).T
        elif values.size % data_size == 0:
            rhs_count = values.size // data_size
            rhs_shape = (rhs_count,) if rhs_count > 1 else ()
            value_block = values.reshape(data_size, rhs_count)
        else:
            raise ValueError(f"Shape {values.shape} incompatible with data_shape {data_shape}.")

        result = operator.matmat(value_block)
        return result.reshape(output_shape + rhs_shape)

    def apply_scalar_regularization(self, coeffs):
        """Apply scalar degree regularization to coefficients."""
        operator = self.scalar_regularization_operator
        if operator is None:
            raise RuntimeError("Scalar regularization requires reg_lambda to be configured.")
        coeff_array = self._coefficient_array(coeffs)
        return operator.matvec(coeff_array)

    def apply_helmholtz_regularization(self, coeffs):
        """Apply Helmholtz degree regularization to coefficients."""
        operator = self.helmholtz_regularization_operator
        if operator is None:
            raise RuntimeError("Helmholtz regularization requires reg_lambda to be configured.")
        coeff_array = self._coefficient_array(coeffs, helmholtz=True)
        return operator.matvec(coeff_array.reshape(-1)).reshape(coeff_array.shape)

    def analyze_scalar_samples(
        self,
        values,
        *,
        input_grid,
        analysis_basis,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Analyze scalar samples and return one coefficient row per batch item."""
        return self._analyze_samples(
            values,
            input_grid=input_grid,
            analysis_basis=analysis_basis,
            helmholtz=False,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def analyze_helmholtz_samples(
        self,
        values,
        *,
        input_grid,
        analysis_basis,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Analyze tangential samples and return one coefficient row per batch item."""
        return self._analyze_samples(
            values,
            input_grid=input_grid,
            analysis_basis=analysis_basis,
            helmholtz=True,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def _analyze_samples(
        self,
        values,
        *,
        input_grid,
        analysis_basis,
        helmholtz,
        sqrt_weights,
        reg_lambda,
        pinv_rtol,
    ):
        """Analyze one scalar or Helmholtz field batch."""
        if helmholtz:
            value_batch = self.normalize_helmholtz_samples(values, input_grid)
        else:
            value_batch = self.normalize_scalar_samples(values, input_grid)
        direct_analysis = isinstance(analysis_basis, SurfaceDifferentialBasis) and not callable(
            getattr(analysis_basis, "scalar_grid_remap_operator", None)
        )

        if direct_analysis:
            if (
                self.basis is not analysis_basis
                and not self.basis.coefficients_are_compatible_with(analysis_basis)
            ):
                raise ValueError(
                    "Direct analysis basis is not coefficient-compatible with the transform basis."
                )
            analysis_transform = self._get_input_transform(
                analysis_basis,
                input_grid,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )
            grid_values = value_batch
        else:
            analysis_transform = self
            grid_values = (
                value_batch
                if input_grid.same_as(self.grid)
                else self._remap_batch_to_grid(value_batch, input_grid, helmholtz=helmholtz)
            )

        if helmholtz:
            coeffs = analysis_transform.analyze_helmholtz(grid_values)
        else:
            coeffs = analysis_transform.analyze_scalar(grid_values)
        return self._analysis_coefficients_to_rows(
            coeffs, batch_size=value_batch.shape[0], helmholtz=helmholtz
        )

    def _analysis_coefficients_to_rows(self, coeffs, *, batch_size, helmholtz):
        """Return analysis coefficients in time-row layout."""
        array = np.asarray(coeffs)
        if not helmholtz and self._scalar_analysis_is_noop():
            return array.reshape(batch_size, -1)
        if batch_size == 1:
            return array.reshape(1, -1)
        return np.moveaxis(array, -1, 0).reshape(batch_size, -1)

    def _scalar_analysis_is_noop(self):
        """Return whether scalar analysis is a no-op."""
        return is_noop_linear_map(
            self.scalar_synthesis_operator,
            input_shape=(self.basis.index_length,),
            output_shape=(self.grid.size,),
        )

    def _coefficient_array(self, coeffs, *, helmholtz=False, preserve_backend=False):
        """Return validated coefficient values."""
        values = getattr(coeffs, "array", coeffs)
        shape = (2, self.basis.index_length) if helmholtz else (self.basis.index_length,)
        expected_size = int(np.prod(shape))
        size = getattr(values, "size", None)
        if size is None:
            array = np.asarray(values)
            size = array.size
        if int(size) != expected_size:
            field_type = "Helmholtz" if helmholtz else "scalar"
            raise ValueError(
                f"{field_type} coefficients have length {int(size)}, expected {expected_size}."
            )
        if preserve_backend:
            xp = get_array_module(values)
            if xp is not np:
                return xp.asarray(values).reshape(shape)
        return np.asarray(values).reshape(shape)

    def _coefficients_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """Transform basis coefficients to grid values."""
        if derivative == "theta":
            operator = self.theta_derivative_operator
        elif derivative == "phi":
            operator = self.phi_derivative_operator
        elif helmholtz:
            operator = self.helmholtz_synthesis_operator
        else:
            operator = self.scalar_synthesis_operator

        return operator.matvec(coeffs).reshape(operator.output_shape)

    @staticmethod
    def normalize_scalar_samples(values, input_grid):
        """Return scalar values with canonical time-first layout."""
        n_points = int(input_grid.size)
        array = np.asarray(values)

        if array.ndim == 1:
            if array.size != n_points:
                raise ValueError(f"Scalar field has {array.size} points, expected {n_points}.")
            return array.reshape(1, n_points)
        if array.ndim == 2:
            if array.shape[-1] == n_points:
                return array
            if array.shape[0] == n_points:
                return array.T
        raise ValueError(
            "Scalar sample analysis expects shape (N,), (B, N), or (N, B); "
            f"got {array.shape} for grid size {n_points}."
        )

    @staticmethod
    def normalize_helmholtz_samples(values, input_grid):
        """Return tangential values with canonical time-first layout."""
        n_points = int(input_grid.size)
        array = np.asarray(values)

        if array.ndim == 2:
            if array.shape == (2, n_points):
                return array.reshape(1, 2, n_points)
            if array.shape == (n_points, 2):
                return array.T.reshape(1, 2, n_points)
        elif array.ndim == 3:
            if array.shape[1:] == (2, n_points):
                return array
            if array.shape[:2] == (2, n_points):
                return np.moveaxis(array, -1, 0)
            if array.shape[1:] == (n_points, 2):
                return np.moveaxis(array, -1, 1)

        raise ValueError(
            "Tangential sample analysis expects shape (2, N), (B, 2, N), "
            f"(N, 2), or (B, N, 2); got {array.shape} for grid size {n_points}."
        )

    def _get_input_transform(
        self, analysis_basis, input_grid, *, sqrt_weights=None, reg_lambda=None, pinv_rtol=1e-15
    ):
        """Return transform for direct input-grid analysis."""
        if sqrt_weights is not None:
            weight_signature = array_fingerprint(sqrt_weights)
            if weight_signature is None:
                return SphericalTransform(
                    analysis_basis,
                    input_grid,
                    sqrt_weights=sqrt_weights,
                    reg_lambda=reg_lambda,
                    pinv_rtol=pinv_rtol,
                    area_weighted=self.area_weighted,
                )
            grid_signature = input_grid.signature
        else:
            weight_signature = None
            grid_signature = (
                input_grid.analysis_signature if self.area_weighted else input_grid.signature
            )

        cache_key = (
            analysis_basis.signature,
            grid_signature,
            weight_signature,
            reg_lambda,
            pinv_rtol,
            self.area_weighted,
        )
        if cache_key in self._input_transforms:
            self._input_transforms.move_to_end(cache_key)
            return self._input_transforms[cache_key]

        transform = SphericalTransform(
            analysis_basis,
            input_grid,
            sqrt_weights=(None if sqrt_weights is None else _owned_explicit_weights(sqrt_weights)),
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
            area_weighted=self.area_weighted,
        )
        self._input_transforms[cache_key] = transform
        if len(self._input_transforms) > self._input_transform_cache_size:
            self._input_transforms.popitem(last=False)
        return transform

    def _grid_remap_operator(self, method_name, input_grid, *, input_shape, output_shape):
        """Return the required grid-remap operator."""
        if self.grid_remap_basis is None:
            raise ValueError("grid_remap_basis is required for grid remapping.")
        remap_operator = getattr(self.grid_remap_basis, method_name, None)
        if not callable(remap_operator):
            raise TypeError(
                f"SphericalGrid-to-grid sample analysis requires grid_remap_basis to provide "
                f"{method_name}()."
            )
        operator = remap_operator(input_grid, self.grid)
        try:
            return as_linear_map(operator, input_shape=input_shape, output_shape=output_shape)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{type(self.grid_remap_basis).__name__}.{method_name}() "
                "must return an operator convertible to LinearMap."
            ) from exc

    def _remap_batch_to_grid(self, value_batch, input_grid, *, helmholtz):
        """Apply grid remap operators to field slices."""
        if not helmholtz:
            operator = self._grid_remap_operator(
                "scalar_grid_remap_operator",
                input_grid,
                input_shape=(input_grid.size,),
                output_shape=(self.grid.size,),
            )
            interpolated = operator.matmat(np.asarray(value_batch).T)
            return np.asarray(interpolated).reshape(self.grid.size, -1).T

        values = np.asarray(value_batch)
        operator = self._grid_remap_operator(
            "tangential_grid_remap_operator",
            input_grid,
            input_shape=(2, input_grid.size),
            output_shape=(2, self.grid.size),
        )
        interpolated = operator.matmat(values.reshape(values.shape[0], -1).T)
        return np.moveaxis(np.asarray(interpolated).reshape(2, self.grid.size, -1), -1, 0)
