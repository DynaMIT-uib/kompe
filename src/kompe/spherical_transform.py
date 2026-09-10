"""Spherical transform module.

This module contains the SphericalTransform class for converting between
spherical-basis coefficients and grid values.
"""

from functools import cached_property

import numpy as np
from scipy.linalg import cholesky

from kompe.basis import ScalarBasis, SurfaceDifferentialBasis
from kompe.cache import BoundedCache
from kompe.grid import SphericalGrid
from kompe.math import array_fingerprint
from kompe.math.backend import _is_jax_tracer, get_array_module, immutable_array
from kompe.math.least_squares_problem import (
    LeastSquaresProblem,
    as_rhs_block,
    relative_regularization,
)
from kompe.math.least_squares_solver import (
    LeastSquaresSolver,
    cholesky_least_squares_map,
    get_default_least_squares_solver,
)
from kompe.math.linear_map import (
    as_linear_map,
    diagonal_linear_map,
    is_identity_linear_map,
)
from kompe.math.pseudoinverse import weighted_tensor_pinv

_LEAST_SQUARES_CACHE_VERSION = 6


def _normalize_regularization_lambda(value):
    """Return a positive regularization strength or ``None``."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("reg_lambda must be a finite non-negative scalar or None.")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("reg_lambda must be a finite non-negative scalar or None.")
    return None if value == 0.0 else value


def _normalize_tolerance(value):
    """Return a finite, non-negative solver tolerance."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("tolerance must be a finite non-negative scalar.")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tolerance must be a finite non-negative scalar.")
    return value


def grid_sqrt_area_weights(grid):
    """Return supplied quadrature weights, never infer a sampling measure."""
    if not hasattr(grid, "area_weights"):
        raise ValueError(
            "Area-weighted analysis requires grid.area_weights or explicit sqrt_weights; "
            "point coordinates alone do not define integration weights."
        )
    xp = get_array_module(grid.area_weights)
    return xp.sqrt(xp.asarray(grid.area_weights, dtype=float))


def resolve_sqrt_weights(grid, sqrt_weights=None, area_weighted=False, vector=False):
    """Resolve explicit or default grid sqrt weights."""
    if sqrt_weights is not None:
        xp = get_array_module(sqrt_weights)
        weights = immutable_array(sqrt_weights)
        if weights.size not in {grid.size, 2 * grid.size}:
            raise ValueError(
                "sqrt_weights must contain one value per point or tangential component."
            )
        if not _is_jax_tracer(weights) and not bool(xp.all(xp.isfinite(weights) & (weights >= 0))):
            raise ValueError("sqrt_weights must be finite and non-negative.")
        if vector and weights.size == grid.size:
            return xp.broadcast_to(weights.reshape(1, grid.size), (2, grid.size))
        return weights
    if not area_weighted:
        return None
    weights = grid_sqrt_area_weights(grid)
    xp = get_array_module(weights)
    return xp.tile(weights, (2, 1)) if vector else weights


class SphericalTransform:
    """Two-way transform between a spherical basis and a grid.

    This class owns both synthesis (coefficients to grid values) and
    analysis (grid values to coefficients) for scalar and tangential
    Helmholtz fields. Numerical arrays have leading component/point or
    coefficient axes and trailing batch axes. External samples can be
    fitted on their own grid or passed through an explicit remapping
    operator before fitting on the bound grid.
    """

    _cached_attribute_names = (
        "scalar_synthesis_array",
        "scalar_synthesis_operator",
        "gradient_theta_array",
        "gradient_theta_operator",
        "gradient_phi_array",
        "gradient_phi_operator",
        "surface_gradient_array",
        "surface_gradient_operator",
        "rhat_cross_gradient_array",
        "rhat_cross_gradient_operator",
        "helmholtz_synthesis_array",
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
        sqrt_weights=None,
        reg_lambda=None,
        tolerance=1e-15,
        area_weighted=False,
        use_persistent_evaluation_cache=True,
    ):
        """Initialize synthesis and analysis between ``basis`` and ``grid``.

        Parameters
        ----------
        basis : ScalarBasis
            Coefficient representation to evaluate or fit. Helmholtz fields
            and surface-smoothness regularization require SurfaceDifferentialBasis.
        grid : SphericalGrid
            Sample positions in the same spherical coordinate frame as the
            basis.
        sqrt_weights : array-like, optional
            Square-root residual weights. Their squares weight the
            least-squares objective. One weight per grid point applies to
            both tangential components; shape ``(2, grid.size)`` supplies
            component-specific weights.
        reg_lambda : float, optional
            Dimensionless relative regularization strength. Kompe balances
            the data and regularization operator scales before solving.
        tolerance : float, optional
            Numerical tolerance for the least-squares solver.
        area_weighted : bool, optional
            Use the supplied ``grid.area_weights``. Coordinates alone do not
            define a quadrature rule. Explicit ``sqrt_weights`` take precedence.
            This affects analysis only, never synthesis.
        use_persistent_evaluation_cache : bool, optional
            Allow basis-evaluation disk-cache reads and writes. False applies
            to scalar and vector evaluations alike; bounded in-memory reuse
            remains available. Numerical fit-factor caching is separate.
        """
        if not isinstance(basis, ScalarBasis):
            raise TypeError("SphericalTransform basis must implement ScalarBasis.")
        if not isinstance(grid, SphericalGrid):
            raise TypeError("SphericalTransform grid must be a SphericalGrid.")
        self.basis = basis
        self.grid = grid
        self.explicit_sqrt_weights = sqrt_weights is not None
        self.area_weighted = bool(area_weighted)
        self.sqrt_weights = resolve_sqrt_weights(
            grid, sqrt_weights=sqrt_weights, area_weighted=area_weighted
        )
        self.helmholtz_sqrt_weights = resolve_sqrt_weights(
            grid, sqrt_weights=self.sqrt_weights, vector=True
        )
        self.reg_lambda = _normalize_regularization_lambda(reg_lambda)
        self.tolerance = _normalize_tolerance(tolerance)
        self.use_persistent_evaluation_cache = bool(use_persistent_evaluation_cache)

        self._analysis_transforms = BoundedCache(16)
        self._basis_transforms = BoundedCache(8)

    def __repr__(self):
        """Summarize the bound basis, grid, and analysis policy."""
        return (
            f"SphericalTransform(basis={self.basis!r}, grid={self.grid!r}, "
            f"area_weighted={self.area_weighted}, reg_lambda={self.reg_lambda!r})"
        )

    def with_basis(self, basis):
        """Return this transform bound to a coefficient basis on the same grid."""
        if not isinstance(basis, ScalarBasis):
            raise TypeError("basis must implement ScalarBasis.")
        if self.basis.signature == basis.signature:
            return self
        cache_key = basis.signature

        def build():
            return SphericalTransform(
                basis,
                self.grid,
                sqrt_weights=self.sqrt_weights if self.explicit_sqrt_weights else None,
                reg_lambda=self.reg_lambda,
                tolerance=self.tolerance,
                area_weighted=self.area_weighted,
                use_persistent_evaluation_cache=self.use_persistent_evaluation_cache,
            )

        return self._basis_transforms.get_or_create(cache_key, build)

    def clear_cache(self):
        """Discard arrays, operators, factorizations, and analysis transforms."""
        for name in self._cached_attribute_names:
            self.__dict__.pop(name, None)
        self._analysis_transforms.clear()
        self._basis_transforms.clear()

    def cache_info(self):
        """Return local transform-cache occupancy and configuration."""
        cached_attributes = sum(name in self.__dict__ for name in self._cached_attribute_names)
        return {
            "cached_attributes": cached_attributes,
            "scalar_problem_cached": "scalar_least_squares_problem" in self.__dict__,
            "helmholtz_problem_cached": "helmholtz_least_squares_problem" in self.__dict__,
            "analysis_transforms": len(self._analysis_transforms),
            "analysis_transform_max_size": self._analysis_transforms.max_size,
            "basis_transforms": len(self._basis_transforms),
            "basis_transform_max_size": self._basis_transforms.max_size,
        }

    def _scalar_evaluation_operator(self, gradient_component=None):
        """Apply the transform's persistent-evaluation policy at the basis boundary."""
        return self.basis.scalar_evaluation_operator(
            self.grid,
            gradient_component=gradient_component,
            persist=self.use_persistent_evaluation_cache,
        )

    @property
    def _surface_basis(self):
        """Require closed-surface semantics only for operations that use them."""
        if not isinstance(self.basis, SurfaceDifferentialBasis):
            raise TypeError(
                "Helmholtz analysis/synthesis and surface-smoothness regularization "
                "require a SurfaceDifferentialBasis; scalar evaluation and fitting do not."
            )
        return self.basis

    def _operator_cache(self):
        """Return the basis's persistent operator cache."""
        return getattr(self.basis.root_basis, "operator_cache", None)

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
            "grid_coordinates": self.grid.signature,
            "sqrt_weights": array_fingerprint(weights),
            "regularization_lambda": self.reg_lambda,
            "area_weighted": self.area_weighted,
        }

    @cached_property
    def scalar_synthesis_array(self):
        """Array mapping scalar coefficients to grid values."""
        return self.scalar_synthesis_operator.to_array()

    @cached_property
    def scalar_synthesis_operator(self):
        """Operator mapping scalar coefficients to grid values."""
        return self._scalar_evaluation_operator()

    @cached_property
    def gradient_theta_array(self):
        """Evaluate ``d/dtheta``, the unit-sphere theta gradient component."""
        gradient = self.__dict__.get("surface_gradient_array")
        return gradient[0] if gradient is not None else self.gradient_theta_operator.to_array()

    @cached_property
    def gradient_theta_operator(self):
        """Evaluate ``d/dtheta``, the unit-sphere theta gradient component."""
        return self._scalar_evaluation_operator("theta")

    @cached_property
    def gradient_phi_array(self):
        """Evaluate ``(1/sin(theta)) d/dphi`` on the unit sphere."""
        gradient = self.__dict__.get("surface_gradient_array")
        return gradient[1] if gradient is not None else self.gradient_phi_operator.to_array()

    @cached_property
    def gradient_phi_operator(self):
        """Evaluate ``(1/sin(theta)) d/dphi`` on the unit sphere."""
        return self._scalar_evaluation_operator("phi")

    @cached_property
    def surface_gradient_array(self):
        """Evaluate the unit-sphere gradient in ``(theta, phi)`` order."""
        return self.surface_gradient_operator.to_array()

    @cached_property
    def surface_gradient_operator(self):
        """Evaluate the unit-sphere gradient in ``(theta, phi)`` order."""
        return self.basis.surface_gradient_operator(
            self.grid, persist=self.use_persistent_evaluation_cache
        )

    @cached_property
    def rhat_cross_gradient_array(self):
        """Array evaluating r-hat x horizontal gradient."""
        return self.rhat_cross_gradient_operator.to_array()

    @cached_property
    def rhat_cross_gradient_operator(self):
        """Operator evaluating r-hat x horizontal gradient."""
        return self.basis.rhat_cross_gradient_operator(
            self.grid, persist=self.use_persistent_evaluation_cache
        )

    @cached_property
    def helmholtz_synthesis_array(self):
        """Array evaluating horizontal vector field expansions."""
        return self.helmholtz_synthesis_operator.to_array()

    @cached_property
    def helmholtz_synthesis_operator(self):
        """Operator evaluating horizontal vector field expansions."""
        return self._surface_basis.helmholtz_synthesis_operator(
            self.grid, persist=self.use_persistent_evaluation_cache
        )

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
        # Use the same physical gauges as configurable analysis. A plain
        # pseudoinverse fixes the Euclidean coefficient mean, which is not
        # the surface-area mean of a nodal basis.
        problem = self.helmholtz_least_squares_problem
        reduced = problem.reduced_problem
        analysis = weighted_tensor_pinv(
            reduced.data_operators[0].to_array(),
            sqrt_weights=self.helmholtz_sqrt_weights,
            output_ndim=2,
            rtol=self.tolerance,
        )
        inverse = as_linear_map(
            analysis,
            input_shape=(2, self.grid.size),
            output_shape=reduced.solution_shape,
        )
        Z = problem.solution_basis
        return inverse if Z is None else Z @ inverse

    @cached_property
    def _optimized_helmholtz_analysis_operator(self):
        """Return an available structured or factorized analysis map."""
        factory = getattr(self._surface_basis, "helmholtz_analysis_operator", None)
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
        if not self.basis.omits_constant_mode():
            return None

        def factor_normal():
            normal = self.helmholtz_synthesis_operator.normal_operator(
                self.helmholtz_sqrt_weights
            ).to_matrix(backend="numpy")
            return cholesky(normal, lower=True, overwrite_a=True, check_finite=False)

        try:
            cache = self._operator_cache()
            identity = self._least_squares_cache_identity("helmholtz")
            if cache is not None and identity is not None:
                factor = cache.get_or_create(
                    "least_squares_factor",
                    {**identity, "factorization": "structured_helmholtz_cholesky"},
                    factor_normal,
                )
            else:
                factor = factor_normal()
            return cholesky_least_squares_map(
                self.helmholtz_synthesis_operator,
                factor,
                sqrt_weights=self.helmholtz_sqrt_weights,
            )
        except np.linalg.LinAlgError:
            return None

    def rhat_cross_gradient_analysis_operator(self, *, coefficient_scale=None):
        """Factor analysis for one rotated-gradient potential."""
        if coefficient_scale is None:
            coefficient_scale = np.ones(self.basis.coefficient_count)
        scale = np.asarray(coefficient_scale)
        synthesis = self.rhat_cross_gradient_operator @ diagonal_linear_map(scale)
        normal = synthesis.normal_operator(self.helmholtz_sqrt_weights).to_matrix(backend="numpy")
        try:
            factor = cholesky(normal, lower=True, overwrite_a=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Rotated-gradient synthesis must have full column rank.") from exc
        return cholesky_least_squares_map(
            synthesis,
            factor,
            sqrt_weights=self.helmholtz_sqrt_weights,
        )

    @cached_property
    def scalar_regularization_operator(self):
        """Surface-gradient smoothness operator for scalar fields."""
        if self.reg_lambda is None:
            return None
        return self._surface_basis.scalar_smoothness_operator()

    @cached_property
    def helmholtz_regularization_operator(self):
        """Return equal-component Helmholtz-field smoothness."""
        if self.reg_lambda is None:
            return None
        return self._surface_basis.helmholtz_smoothness_operator()

    @cached_property
    def scalar_least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for scalar fields."""
        if self.sqrt_weights is not None and self.sqrt_weights.size != self.grid.size:
            raise ValueError(
                "Component-specific sqrt_weights define only a Helmholtz fit. "
                "Scalar fitting requires one weight per grid point."
            )
        return LeastSquaresProblem(
            A=self.scalar_synthesis_operator,
            sqrt_weights=self.sqrt_weights,
            regularization=relative_regularization(
                self.scalar_synthesis_operator,
                self.scalar_regularization_operator,
                self.reg_lambda,
                sqrt_weights=self.sqrt_weights,
            ),
            operator_cache=self._operator_cache(),
            cache_identity=self._least_squares_cache_identity("scalar"),
        )

    @cached_property
    def helmholtz_least_squares_problem(self) -> LeastSquaresProblem:
        """Fit vector samples with zero-mean gauges for constant potentials.

        Constrain a potential's mean only if the basis represents constants;
        otherwise that constraint would change the observable field.
        """
        return LeastSquaresProblem(
            A=self.helmholtz_synthesis_operator,
            constraints=self.basis.helmholtz_gauge_constraints,
            sqrt_weights=self.helmholtz_sqrt_weights,
            regularization=relative_regularization(
                self.helmholtz_synthesis_operator,
                self.helmholtz_regularization_operator,
                self.reg_lambda,
                sqrt_weights=self.helmholtz_sqrt_weights,
            ),
            operator_cache=self._operator_cache(),
            cache_identity=self._least_squares_cache_identity("helmholtz"),
        )

    def _solve_least_squares(self, problem, rhs, solver=None):
        """Apply the requested algorithm, with no analysis-specific dispatch."""
        if not isinstance(solver, LeastSquaresSolver):
            solver = LeastSquaresSolver(method=solver, tolerance=self.tolerance)
        return solver.solve(problem, rhs)

    def synthesize_scalar(self, coeffs, gradient_component=None):
        """Synthesize scalar values or a unit-sphere gradient component.

        ``gradient_component`` is ``None``, ``'theta'`` for ``d/dtheta``,
        or ``'phi'`` for ``(1/sin(theta)) d/dphi``. Batch axes trail.
        """
        coeff_array = self._coefficient_array(coeffs)
        if gradient_component is None:
            operator = self.scalar_synthesis_operator
        elif gradient_component == "theta":
            operator = self.gradient_theta_operator
        elif gradient_component == "phi":
            operator = self.gradient_phi_operator
        else:
            raise ValueError("gradient_component must be None, 'theta', or 'phi'.")
        return operator(coeff_array)

    def synthesize_helmholtz(self, coeffs):
        """Synthesize tangential values, retaining trailing batch axes."""
        coeff_array = self._coefficient_array(coeffs, helmholtz=True)
        operator = self.helmholtz_synthesis_operator
        return operator(coeff_array)

    def analyze_scalar(self, grid_values, solver=None):
        """Analyze scalar values, returning ``(n_coeffs, *batch_shape)``.

        Values have shape ``(n_points, *batch_shape)``.
        An unbatched field returns a one-dimensional coefficient array.
        ``solver`` is a method name or a reusable ``LeastSquaresSolver``.
        Native scalar nodal values need no fit when unregularized and
        unweighted; that identity operation is returned directly.
        """
        if (
            self._scalar_synthesis_is_identity()
            and self.reg_lambda is None
            and not self.explicit_sqrt_weights
        ):
            values, batch_shape = as_rhs_block(grid_values, (self.grid.size,))
            return values.reshape((self.basis.coefficient_count,) + batch_shape)
        values = get_array_module(grid_values).asarray(grid_values)
        return self._solve_least_squares(self.scalar_least_squares_problem, values, solver)

    def analyze_helmholtz(self, grid_values, solver=None):
        """Analyze tangential values, returning ``(2, n_coeffs, *batch_shape)``.

        Data axes ``(2, n_points)`` precede batch axes. A flat single field is also
        accepted. Components are ordered ``(theta, phi)``.
        ``solver`` is a method name or a reusable ``LeastSquaresSolver``.
        It defaults to ``KOMPE_LEAST_SQUARES_SOLVER`` (``normal_pinv``).
        ``normal_solve`` reuses a sparse or Cholesky factorization when
        available for this same unregularized objective. Other methods
        keep their requested cutoff or iterative convergence behavior.
        ``helmholtz_analysis_operator`` exposes the fixed inverse itself.
        """
        values = get_array_module(grid_values).asarray(grid_values)
        if isinstance(solver, LeastSquaresSolver):
            method = solver.method
        else:
            method = get_default_least_squares_solver() if solver is None else solver
        if (
            method == "normal_solve"
            and self.reg_lambda is None
            and (not isinstance(solver, LeastSquaresSolver) or solver.preconditioner_type is None)
        ):
            inverse = self._optimized_helmholtz_analysis_operator
            if inverse is not None:
                block, batch_shape = as_rhs_block(values, (2, self.grid.size))
                return inverse.matmat(block).reshape(
                    (2, self.basis.coefficient_count) + batch_shape
                )
        return self._solve_least_squares(self.helmholtz_least_squares_problem, values, solver)

    def apply_scalar_regularization(self, coeffs):
        """Apply the unscaled scalar smoothness operator to coefficients."""
        operator = self.scalar_regularization_operator
        if operator is None:
            raise RuntimeError("Scalar regularization requires reg_lambda to be configured.")
        coeff_array = self._coefficient_array(coeffs)
        return operator(coeff_array)

    def apply_helmholtz_regularization(self, coeffs):
        """Apply the unscaled Helmholtz smoothness operator to coefficients."""
        operator = self.helmholtz_regularization_operator
        if operator is None:
            raise RuntimeError("Helmholtz regularization requires reg_lambda to be configured.")
        coeff_array = self._coefficient_array(coeffs, helmholtz=True)
        return operator(coeff_array)

    def analyze_scalar_samples(
        self,
        values,
        *,
        input_grid,
        remap=None,
        sqrt_weights=None,
        reg_lambda=None,
        solver=None,
    ):
        """Fit ``(input_grid.size, *batch_shape)`` samples to this basis.

        Without ``remap``, fit directly on ``input_grid``. An explicit
        scalar ``LinearMap`` from ``input_grid`` to ``self.grid`` instead
        remaps values before fitting on the bound grid. Returned arrays
        have shape ``(n_coeffs, *batch_shape)``, ready for synthesis.

        ``reg_lambda=None`` inherits this transform's regularization; zero
        disables it. Omitted ``sqrt_weights`` inherit explicit transform
        weights only on the same grid. For direct fits on another grid,
        supply its weights or use ``area_weighted``. Remapped samples use
        the target transform's weights; source weights cannot be remapped.
        """
        return self._analyze_samples(
            values,
            input_grid=input_grid,
            remap=remap,
            helmholtz=False,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            solver=solver,
        )

    def analyze_helmholtz_samples(
        self,
        values,
        *,
        input_grid,
        remap=None,
        sqrt_weights=None,
        reg_lambda=None,
        solver=None,
    ):
        """Fit tangential samples, retaining trailing batch axes.

        Input shape is ``(2, input_grid.size, *batch_shape)`` with
        ``(theta, phi)`` components. Output shape is
        ``(2, n_coeffs, *batch_shape)`` with curl-free then divergence-free
        potentials. An optional tangential ``remap`` maps input samples
        to ``self.grid`` before fitting; otherwise fit on ``input_grid``.

        Regularization and weight defaults follow ``analyze_scalar_samples``:
        inherit regularization, and keep explicit weights on their own grid.
        Pass ``reg_lambda=0`` to disable regularization.
        """
        return self._analyze_samples(
            values,
            input_grid=input_grid,
            remap=remap,
            helmholtz=True,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            solver=solver,
        )

    def _analyze_samples(
        self,
        values,
        *,
        input_grid,
        remap,
        helmholtz,
        sqrt_weights,
        reg_lambda,
        solver,
    ):
        """Remap when requested, then fit with the appropriate grid's weights."""
        if not isinstance(input_grid, SphericalGrid):
            raise TypeError("input_grid must be a SphericalGrid.")
        same_grid = input_grid.same_as(self.grid)
        reg_lambda = _normalize_regularization_lambda(
            self.reg_lambda if reg_lambda is None else reg_lambda
        )
        fit_grid = input_grid
        if remap is not None:
            if sqrt_weights is not None and not same_grid:
                raise ValueError(
                    "sqrt_weights describe the input samples and cannot be propagated through "
                    "grid remapping; configure target-grid weights on SphericalTransform instead."
                )
            components = (2,) if helmholtz else ()
            operator = as_linear_map(
                remap,
                input_shape=components + (input_grid.size,),
                output_shape=components + (self.grid.size,),
            )
            block, batch_shape = as_rhs_block(values, operator.input_shape)
            values = operator.matmat(block).reshape(operator.output_shape + batch_shape)
            fit_grid = self.grid

        if (
            sqrt_weights is None
            and self.explicit_sqrt_weights
            and (same_grid or remap is not None)
        ):
            sqrt_weights = self.sqrt_weights
        transform = self._sample_analysis_transform(
            fit_grid, sqrt_weights=sqrt_weights, reg_lambda=reg_lambda
        )
        analyze = transform.analyze_helmholtz if helmholtz else transform.analyze_scalar
        return analyze(values, solver=solver)

    def _scalar_synthesis_is_identity(self):
        """Return whether scalar analysis is a no-op."""
        return is_identity_linear_map(
            self.scalar_synthesis_operator,
            input_shape=(self.basis.coefficient_count,),
            output_shape=(self.grid.size,),
        )

    def _coefficient_array(self, coeffs, *, helmholtz=False):
        """Normalize flat single-field inputs without changing batch axes."""
        shape = (2, self.basis.coefficient_count) if helmholtz else (self.basis.coefficient_count,)
        expected_size = int(np.prod(shape))
        xp = get_array_module(coeffs)
        array = xp.asarray(coeffs)
        if array.shape[: len(shape)] != shape and array.size == expected_size:
            return array.reshape(shape)
        return array

    def _sample_analysis_transform(
        self,
        input_grid,
        *,
        sqrt_weights=None,
        reg_lambda=None,
    ):
        """Reuse the bound fit or a cached fit on another grid or with other settings."""
        use_grid_measure = self.area_weighted and sqrt_weights is None
        grid_signature = (
            input_grid.analysis_signature if use_grid_measure else input_grid.signature
        )
        bound_grid_signature = (
            self.grid.analysis_signature if use_grid_measure else self.grid.signature
        )
        same_setup = grid_signature == bound_grid_signature and reg_lambda == self.reg_lambda
        bound_weights = self.sqrt_weights if self.explicit_sqrt_weights else None
        # Inherited weights need no hashing or device-to-host transfer.
        if same_setup and sqrt_weights is bound_weights:
            return self
        weight_signature = None if sqrt_weights is None else array_fingerprint(sqrt_weights)
        if (
            same_setup
            and weight_signature is not None
            and weight_signature == array_fingerprint(bound_weights)
        ):
            return self

        def build():
            return SphericalTransform(
                self.basis,
                input_grid,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                tolerance=self.tolerance,
                area_weighted=self.area_weighted,
                use_persistent_evaluation_cache=self.use_persistent_evaluation_cache,
            )

        # Traced weights belong to this call, not the persistent Python cache.
        if sqrt_weights is not None and weight_signature is None:
            return build()
        cache_key = (
            grid_signature,
            weight_signature,
            reg_lambda,
            self.area_weighted,
        )
        return self._analysis_transforms.get_or_create(cache_key, build)
