"""Tests for least-squares solver helpers."""

import warnings

import numpy as np
import pytest
from scipy.sparse.linalg import lsmr as scipy_lsmr

from kompe.math import (
    LinearMap,
    as_linear_map,
    backend_context,
    get_array_module,
    jax_enabled,
    relative_regularization,
    set_backend,
)
from kompe.math.least_squares_problem import LeastSquaresProblem
from kompe.math.least_squares_solver import (
    LEAST_SQUARES_SOLVER_ENV,
    LeastSquaresSolver,
    dense_full_rank_least_squares_map,
    get_default_least_squares_solver,
    sparse_least_squares_map,
)

# Problem definition and regularization


def _with_normal_product(operator, builder):
    """Declare a fixed test objective's known normal product on its map."""
    if builder is None:
        return operator
    operator = as_linear_map(operator)
    return LinearMap(
        shape=operator.shape,
        dtype=operator.dtype,
        matvec=operator.matvec,
        rmatvec=operator.rmatvec,
        matmat=operator.matmat,
        rmatmat=operator.rmatmat,
        dense_array=operator._dense_array,
        normal_matrix=lambda xp, row_scale: builder(),
        normal_matrix_diag=operator.normal_matrix_diag,
        backend_operands=operator.backend_operands,
        input_shape=operator.input_shape,
        output_shape=operator.output_shape,
    )


@pytest.mark.parametrize("kind", ["dense", "sparse", "diagonal", "matrix_free", "composed"])
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd", "lsmr", "cgls"])
def test_shaped_problem_works_with_all_operator_representations(kind, materialized, solver_name):
    """Shapes are independent of backend and numerical representation."""
    from scipy.sparse import csr_matrix

    from kompe.math import diagonal_linear_map

    xp = get_array_module()
    diagonal = xp.arange(1.0, 5.0)
    matrix = xp.diag(diagonal)
    if kind == "matrix_free":
        operator = LinearMap(
            shape=(4, 4),
            dtype=matrix.dtype,
            matvec=lambda x: matrix @ x,
            rmatvec=lambda x: matrix.T @ x,
            matmat=lambda x: matrix @ x,
            rmatmat=lambda x: matrix.T @ x,
            backend_operands=(matrix,),
            input_shape=(2, 2),
            output_shape=(4,),
        )
    else:
        values = (
            csr_matrix(np.asarray(matrix))
            if kind == "sparse"
            else diagonal
            if kind == "diagonal"
            else matrix
        )
        operator = as_linear_map(values, input_shape=(2, 2), output_shape=(4,))
        if kind == "composed":
            operator = diagonal_linear_map(xp.ones(4)) @ operator
    if materialized:
        operator.to_matrix()
    problem = LeastSquaresProblem(operator)
    assert problem.data_operators[0] is operator
    assert bool(operator._dense_cache) == materialized
    coefficients = xp.arange(1.0, 25.0).reshape(2, 2, 3, 2)
    result = LeastSquaresSolver(method=solver_name, tolerance=1e-13).solve(
        problem, operator(coefficients)
    )
    assert result.shape == coefficients.shape
    np.testing.assert_allclose(result, coefficients, rtol=1e-11, atol=1e-11)
    if solver_name in {"lsmr", "cgls"} and not materialized:
        assert not operator._dense_cache


@pytest.mark.parametrize("sparse", [False, True])
def test_problem_defaults_to_matrix_axes(sparse):
    """Raw matrices retain their row and column dimensions."""
    from scipy.sparse import csr_matrix

    matrix = np.arange(12.0).reshape(4, 3)
    problem = LeastSquaresProblem(csr_matrix(matrix) if sparse else matrix)

    assert problem.solution_shape == (3,)
    assert problem.data_shapes == [(4,)]
    np.testing.assert_array_equal(problem.data_operators[0].to_matrix(), matrix)


def test_problem_reads_map_shapes_without_evaluating_operators():
    """Shape metadata requires neither application nor materialization."""

    def unexpected_evaluation(*args):
        pytest.fail("Problem construction must not evaluate the operator.")

    operators = [
        LinearMap(
            shape=(size, 4),
            dtype=float,
            input_shape=(2, 2),
            output_shape=shape,
            matvec=unexpected_evaluation,
            rmatvec=unexpected_evaluation,
            dense_array=unexpected_evaluation,
        )
        for size, shape in [(6, (2, 3)), (4, (4,))]
    ]
    problem = LeastSquaresProblem(operators)

    assert problem.solution_shape == (2, 2)
    assert problem.solution_size == 4
    assert problem.data_shapes == [(2, 3), (4,)]
    assert all(
        actual is original
        for actual, original in zip(problem.data_operators, operators, strict=True)
    )


@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd", "lsmr", "cgls"])
def test_inferred_problem_shapes_preserve_batched_solutions(solver_name):
    """Different data axes can constrain one shaped coefficient field."""
    xp = get_array_module()
    matrix = xp.asarray(np.vstack([np.eye(4), np.ones((2, 4))]))
    operators = [
        as_linear_map(matrix, input_shape=(2, 2), output_shape=(2, 3)),
        as_linear_map(xp.eye(4), input_shape=(2, 2)),
    ]
    coefficients = xp.arange(24.0).reshape(2, 2, 3, 2)
    rhs = [operator(coefficients) for operator in operators]
    problem = LeastSquaresProblem(operators)

    result = LeastSquaresSolver(method=solver_name, tolerance=1e-13).solve(problem, rhs)

    assert result.shape == coefficients.shape
    np.testing.assert_allclose(result, coefficients, rtol=1e-11, atol=1e-11)


def test_problem_requires_a_shared_coefficient_shape():
    """Equal sizes alone do not make distinct field axes equivalent."""
    operators = [
        as_linear_map(np.eye(4), input_shape=(2, 2)),
        as_linear_map(np.eye(4)),
    ]
    with pytest.raises(ValueError, match="must share input_shape"):
        LeastSquaresProblem(operators)

    problem = LeastSquaresProblem([as_linear_map(op, input_shape=(2, 2)) for op in operators])
    assert all(operator.input_shape == (2, 2) for operator in problem.data_operators)


@pytest.mark.parametrize("data_shapes", [None, (20,)])
def test_problem_can_label_raw_tensor_axes(data_shapes):
    """Explicit input axes determine the tensor's domain boundary."""
    array = np.arange(120.0).reshape(4, 5, 2, 3)
    problem = LeastSquaresProblem(
        as_linear_map(array, input_shape=(2, 3), output_shape=data_shapes)
    )

    assert problem.solution_shape == (2, 3)
    assert problem.data_shapes == [(4, 5) if data_shapes is None else data_shapes]
    np.testing.assert_array_equal(problem.data_operators[0].to_matrix(), array.reshape(20, 6))


def test_problem_retains_scalar_coefficient_shape():
    """An empty shape means one scalar, not a missing shape."""
    operator = as_linear_map(np.array([[2.0], [3.0]]), input_shape=())
    problem = LeastSquaresProblem(operator)

    result = LeastSquaresSolver(method="normal_solve").solve(problem, np.array([4.0, 6.0]))

    assert problem.solution_shape == ()
    assert result.shape == ()
    np.testing.assert_allclose(result, 2.0)


def test_default_solver_reads_only_the_canonical_environment(monkeypatch):
    """Consumer-specific environment names do not affect Kompe."""
    monkeypatch.delenv(LEAST_SQUARES_SOLVER_ENV, raising=False)
    monkeypatch.setenv("PYNAMIT_LEAST_SQUARES_SOLVER", "lsmr")
    assert get_default_least_squares_solver() == "normal_pinv"
    original = LeastSquaresSolver()
    assert original.method == "normal_pinv"

    monkeypatch.setenv(LEAST_SQUARES_SOLVER_ENV, "cgls")
    assert get_default_least_squares_solver() == "cgls"
    assert LeastSquaresSolver().method == "cgls"
    assert LeastSquaresSolver(method=None).method == "cgls"
    assert LeastSquaresSolver("svd").method == "svd"
    assert original.method == "normal_pinv"


def test_explicit_solver_does_not_depend_on_the_environment(monkeypatch):
    """Invalid defaults fail when used, but cannot override an explicit algorithm."""
    monkeypatch.setenv(LEAST_SQUARES_SOLVER_ENV, "not-a-solver")
    with pytest.raises(ValueError, match="Solver must be one of"):
        LeastSquaresSolver()
    assert LeastSquaresSolver("normal_solve").method == "normal_solve"
    with pytest.raises(ValueError, match="Solver must be one of"):
        LeastSquaresSolver("")


def test_unregularized_problem_skips_normal_diagonal_scaling():
    """No-reg problems skip a potentially expensive normal diagonal."""
    problem = LeastSquaresProblem(A=np.eye(2))

    assert problem.regularization_operators == []
    assert "data_operator" not in problem.__dict__

    zero_weight_problem = LeastSquaresProblem(
        A=np.eye(2),
        regularization=relative_regularization(np.eye(2), np.eye(2), 0.0),
    )
    assert zero_weight_problem.regularization_operators == []
    assert "data_operator" not in zero_weight_problem.__dict__


@pytest.mark.parametrize(
    ("regularization_matrix", "regularization_weight"),
    [(np.eye(2), 1e-30), (1e-20 * np.eye(2), 1e-40)],
)
def test_positive_regularization_is_never_silently_discarded(
    regularization_matrix, regularization_weight
):
    """Every positive requested regularization term remains in the objective."""
    problem = LeastSquaresProblem(
        A=np.eye(2),
        regularization=relative_regularization(
            np.eye(2), regularization_matrix, regularization_weight
        ),
    )

    [penalty] = problem.regularization_operators
    assert np.all(np.diag(penalty.to_matrix()) > 0.0)
    assert problem.system_operator.shape == (4, 2)


def test_positive_strength_on_zero_regularization_operator_is_explicit():
    """A requested no-op regularizer should not look as though it was applied."""
    with pytest.raises(ValueError, match="nonzero regularization operator"):
        relative_regularization(np.eye(2), np.zeros((2, 2)), 1.0)


# Explicit least-squares maps


def test_problem_exposes_operators_with_their_mathematical_roles():
    """Stored weighting maps are operators, not raw sqrt-weight values."""
    problem = LeastSquaresProblem(A=np.eye(2), sqrt_weights=np.array([2.0, 3.0]))
    assert isinstance(problem.data_operators[0], LinearMap)
    assert problem.weight_operators[0].is_diagonal
    np.testing.assert_array_equal(problem.weight_operators[0].diagonal(), [2.0, 3.0])
    np.testing.assert_array_equal(problem.data_operator.to_matrix(), np.diag([2.0, 3.0]))


def test_least_squares_requires_a_data_operator():
    """An empty objective is rejected at construction."""
    with pytest.raises(ValueError, match="At least one data operator"):
        LeastSquaresProblem(A=[])


def test_positive_strength_requires_a_regularization_operator():
    """A missing regularizer must not silently disable its strength."""
    with pytest.raises(ValueError, match="requires a regularization operator"):
        relative_regularization(np.eye(2), None, 1.0)


@pytest.mark.parametrize("zero_weights", [False, True])
def test_relative_regularization_requires_a_nonzero_data_scale(zero_weights):
    """Relative strengths cannot be normalized to an arbitrary unit scale."""
    with pytest.raises(ValueError, match="nonzero weighted data operator"):
        relative_regularization(
            np.eye(2) if zero_weights else np.zeros((2, 2)),
            np.eye(2),
            1.0,
            sqrt_weights=np.zeros(2) if zero_weights else None,
        )


@pytest.mark.parametrize("complex_data", [False, True])
def test_sparse_least_squares_map_matches_kkt_and_adjoint(complex_data):
    """Constrained analysis handles rectangular and complex RHS data."""
    A = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
            [1.0, -1.0, 0.0],
            [0.5, 0.25, 1.0],
        ]
    )
    if complex_data:
        A = A + 0.1j * np.array(
            [
                [0.0, 1.0, -0.5],
                [0.5, 0.0, 0.25],
                [-1.0, 0.5, 0.0],
                [0.25, -0.5, 1.0],
            ]
        )
    constraint = np.array([[1.0, 1.0, 1.0]])
    sqrt_weights = np.array([1.0, 2.0, 0.5, 1.5])
    operator = sparse_least_squares_map(
        A,
        constraint,
        sqrt_weights=sqrt_weights,
        input_shape=(2, 2),
        output_shape=(3,),
    )
    rhs = np.array([1.0 + 0.5j, -0.25j, 2.0 - 0.75j, -1.0 + 0.25j])

    weights = np.diag(sqrt_weights**2)
    kkt = np.block(
        [
            [A.T.conjugate() @ weights @ A, constraint.T],
            [constraint, np.zeros((1, 1))],
        ]
    )
    expected_rhs = np.concatenate([A.T.conjugate() @ weights @ rhs, np.zeros(1)])
    expected = np.linalg.solve(kkt, expected_rhs)[: A.shape[1]]
    analysis_rhs = np.vstack(
        [A.T.conjugate() @ weights, np.zeros((constraint.shape[0], A.shape[0]))]
    )
    analysis_matrix = np.linalg.solve(kkt, analysis_rhs)[: A.shape[1]]

    np.testing.assert_allclose(operator.matvec(rhs), expected)
    np.testing.assert_allclose(constraint @ operator.matvec(rhs), np.zeros(1), atol=1e-14)
    coefficient_probe = np.array([0.5 - 0.25j, 1.0j, -0.75 + 0.1j])
    expected_adjoint = analysis_matrix.T.conjugate() @ coefficient_probe
    np.testing.assert_allclose(operator.rmatvec(coefficient_probe), expected_adjoint)
    np.testing.assert_allclose(
        np.vdot(coefficient_probe, operator.matvec(rhs)),
        np.vdot(operator.rmatvec(coefficient_probe), rhs),
        rtol=1e-13,
        atol=1e-13,
    )

    if jax_enabled():
        import jax
        import jax.numpy as jnp

        compiled = jax.jit(operator.matvec)(jnp.asarray(rhs))
        compiled_adjoint = jax.jit(operator.rmatvec)(jnp.asarray(coefficient_probe))
        np.testing.assert_allclose(compiled, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(compiled_adjoint, expected_adjoint, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("kind", ["dense", "sparse", "matrix_free"])
@pytest.mark.parametrize("materialized", [False, True])
def test_cholesky_analysis_inherits_axes_and_preserves_structured_action(kind, materialized):
    """Analysis reverses synthesis axes without needing a dense representation."""
    from scipy.sparse import csr_matrix

    from kompe.math import cholesky_least_squares_map

    xp = get_array_module()
    rng = np.random.default_rng(72)
    matrix = rng.normal(size=(6, 4)) + 1j * rng.normal(size=(6, 4))
    weights = np.linspace(0.5, 1.5, 6)
    factor = np.linalg.cholesky(matrix.T.conj() @ (weights[:, None] ** 2 * matrix))
    if kind == "matrix_free":
        values = xp.asarray(matrix)
        synthesis = LinearMap(
            shape=matrix.shape,
            dtype=matrix.dtype,
            input_shape=(2, 2),
            output_shape=(2, 3),
            matvec=lambda x: values @ x,
            rmatvec=lambda y: values.T.conj() @ y,
            backend_operands=(values,),
        )
    else:
        synthesis = as_linear_map(
            csr_matrix(matrix) if kind == "sparse" else xp.asarray(matrix),
            input_shape=(2, 2),
            output_shape=(2, 3),
        )
    if materialized:
        synthesis.to_matrix()
    analysis = cholesky_least_squares_map(synthesis, factor, sqrt_weights=weights)
    assert analysis.input_shape == (2, 3)
    assert analysis.output_shape == (2, 2)
    assert bool(synthesis._dense_cache) == materialized
    rhs = xp.asarray(rng.normal(size=(2, 3, 2, 3)) + 1j * rng.normal(size=(2, 3, 2, 3)))
    expected_matrix = np.linalg.solve(
        matrix.T.conj() @ (weights[:, None] ** 2 * matrix),
        matrix.T.conj() * weights**2,
    )
    expected = (expected_matrix @ np.asarray(rhs).reshape(6, 6)).reshape(2, 2, 2, 3)
    result = analysis(rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        analysis.adjoint()(result),
        (expected_matrix.T.conj() @ expected.reshape(4, 6)).reshape(rhs.shape),
        rtol=1e-12,
        atol=1e-12,
    )
    assert bool(synthesis._dense_cache) == materialized
    analysis.to_matrix()
    np.testing.assert_allclose(analysis(rhs), expected, rtol=1e-12, atol=1e-12)


def test_dense_full_rank_least_squares_map_matches_weighted_lstsq_and_adjoint():
    """Factorized analysis preserves weighted solutions and adjoints."""
    data_matrix = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
            [1.0, -1.0, 0.0],
            [0.5, 0.25, 1.0],
            [-0.5, 0.75, 0.25],
        ]
    )
    sqrt_weights = np.array([1.0, 2.0, 0.5, 1.5, 0.75])
    operator = dense_full_rank_least_squares_map(
        data_matrix, sqrt_weights=sqrt_weights, input_shape=(5,), output_shape=(3,)
    )
    rhs = np.array([1.0 + 0.5j, -0.25j, 2.0 - 0.75j, -1.0 + 0.25j, 0.5j])

    expected = np.linalg.lstsq(
        sqrt_weights.reshape(-1, 1) * data_matrix, sqrt_weights * rhs, rcond=None
    )[0]
    np.testing.assert_allclose(operator.matvec(rhs), expected, rtol=1e-13, atol=1e-13)

    coefficient_probe = np.array([0.5 - 0.25j, 1.0j, -0.75 + 0.1j])
    np.testing.assert_allclose(
        np.vdot(coefficient_probe, operator.matvec(rhs)),
        np.vdot(operator.rmatvec(coefficient_probe), rhs),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.requires_jax
@pytest.mark.parametrize("complex_rhs", [False, True])
def test_dense_full_rank_least_squares_map_is_jittable_with_jax(complex_rhs):
    """Factorized dense analysis preserves the runtime array backend."""
    import jax
    import jax.numpy as jnp

    data_matrix = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0], [0.5, 0.25]])
    operator = dense_full_rank_least_squares_map(data_matrix)
    rhs = jnp.array([1.0, -0.5, 2.0, 0.25])
    if complex_rhs:
        rhs = rhs * (1.0 + 0.5j)

    actual = jax.jit(operator.matvec)(rhs)
    expected = np.linalg.lstsq(data_matrix, np.asarray(rhs), rcond=None)[0]

    assert "jax" in type(actual).__module__
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    "solver_name, expected", [("svd", [1.0, 1.0]), ("normal_pinv", [1.0, 0.0])]
)
def test_spectral_tolerances_apply_to_the_selected_matrix(solver_name, expected):
    """Normal-pinv retains its cutoff on squared singular values."""
    matrix = np.diag([1.0, 1e-5])
    problem = LeastSquaresProblem(matrix)
    result = LeastSquaresSolver(solver_name, tolerance=1e-8).solve(problem, matrix @ np.ones(2))
    np.testing.assert_allclose(result, expected)


# SciPy LSMR diagnostics


@pytest.mark.parametrize("stop_code", [0, 1, 2])
def test_lsmr_configured_tolerance_stop_codes_do_not_warn(stop_code):
    """LSMR termination at a configured tolerance is quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LeastSquaresSolver._warn_if_lsmr_not_converged(stop_code, column=0)


@pytest.mark.parametrize("stop_code", [4, 5])
def test_lsmr_machine_precision_stop_codes_warn_precisely(stop_code):
    """A machine-precision limit remains visible and precise."""
    with pytest.warns(RuntimeWarning, match="reached machine precision"):
        LeastSquaresSolver._warn_if_lsmr_not_converged(stop_code, column=0)


@pytest.mark.parametrize("stop_code", [3, 6, 7])
def test_lsmr_limit_stop_codes_warn(stop_code):
    """Condition and iteration limit termination remains visible."""
    with pytest.warns(RuntimeWarning, match=rf"stop_code={stop_code}"):
        LeastSquaresSolver._warn_if_lsmr_not_converged(stop_code, column=0)


# Dense solvers and reusable factorizations


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
@pytest.mark.parametrize("custom_builder", [False, True])
def test_normal_solve_reuses_lu_without_retaining_regularized_assembly(
    backend, custom_builder, monkeypatch
):
    """Changing data reuses the direct solve, without augmented zero RHS rows."""
    from scipy import linalg

    if backend == "jax":
        from jax.scipy import linalg

    matrix = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, -2.0]])
    weights = np.array([1.0, 0.5, 2.0])
    penalty = np.array([1.0, 2.0])
    weighted = weights[:, None] * matrix
    row_scale = np.sqrt(0.1 * np.median(np.sum(weighted**2, axis=0)) / np.median(penalty**2))
    augmented = np.vstack([weighted, row_scale * np.diag(penalty)])
    rhs = np.array([[1.0, 0.5], [2.0, -0.5], [1.0, 0.25]])
    expected = np.linalg.lstsq(
        augmented, np.vstack([weights[:, None] * rhs, np.zeros((2, 2))]), rcond=None
    )[0]
    factor = linalg.lu_factor
    calls = []

    def counted_factor(*args, **kwargs):
        calls.append(args[0])
        return factor(*args, **kwargs)

    monkeypatch.setattr(linalg, "lu_factor", counted_factor)
    with backend_context(backend):
        xp = get_array_module()
        problem = LeastSquaresProblem(
            _with_normal_product(
                xp.asarray(matrix), (lambda: weighted.T @ weighted) if custom_builder else None
            ),
            sqrt_weights=weights,
            regularization=relative_regularization(
                xp.asarray(matrix), penalty, 0.1, sqrt_weights=weights
            ),
        )
        solver = LeastSquaresSolver("normal_solve")
        np.testing.assert_allclose(solver.solve(problem, xp.asarray(rhs)), expected, atol=1e-13)
        assert len(calls) == 1

        def fail_rebuild(*_args, **_kwargs):
            pytest.fail("A repeated direct solve must reuse its factorization.")

        monkeypatch.setattr(problem, "dense_normal_matrix", fail_rebuild)
        monkeypatch.setattr(problem, "system_matrix", fail_rebuild)
        result = solver.prepare(problem)(xp.asarray(2 * rhs))
        assert isinstance(result, xp.ndarray)
        np.testing.assert_allclose(result, 2 * expected, atol=1e-13)
        assert len(calls) == 1
        assert not problem.system_operator._dense_cache


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_normal_solve_promotes_factor_precision_for_each_rhs_dtype(backend):
    """A cached single-precision LU must not weaken a later double solve."""
    matrix = np.array([[1.0, 0.7], [1.0, 0.71], [1.0, 0.69]], dtype=np.float32)
    with backend_context(backend):
        xp = get_array_module()
        problem = LeastSquaresProblem(xp.asarray(matrix))
        solver = LeastSquaresSolver("normal_solve")
        normal = np.asarray(problem.dense_normal_matrix(backend=backend))
        for dtype in (np.float32, np.float64, np.complex128, np.float32):
            rhs = xp.asarray([1.0, 2.0, -1.0], dtype=dtype)
            if np.issubdtype(dtype, np.complexfloating):
                rhs = rhs * (1.0 + 0.5j)
            normal_rhs = xp.asarray(matrix).T @ rhs
            expected = np.linalg.solve(normal, np.asarray(normal_rhs))
            result = solver.solve(problem, rhs)
            assert result.dtype == xp.result_type(normal.dtype, rhs.dtype)
            tolerance = 2e-3 if dtype == np.float32 else 2e-11
            np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
@pytest.mark.parametrize("penalty_dtype", [np.float64, np.complex128])
def test_normal_solve_preserves_augmented_system_precision(backend, penalty_dtype):
    """A double-precision penalty must also promote the data product."""
    with backend_context(backend):
        xp = get_array_module()
        matrix = xp.asarray([[1.0, 0.7], [1.0, 0.71], [1.0, 0.69]], dtype=xp.float32)
        rhs = xp.asarray([1.0, 2.0, -1.0], dtype=xp.float32)
        penalty = xp.eye(2, dtype=penalty_dtype)
        if np.issubdtype(penalty_dtype, np.complexfloating):
            penalty = penalty + xp.asarray([[0.0, 0.5j], [0.0, 0.0]])
        problem = LeastSquaresProblem(
            matrix,
            regularization=relative_regularization(matrix, penalty, 0.1),
        )
        augmented = problem.system_matrix()
        augmented_rhs = xp.concatenate([rhs, xp.zeros(2, dtype=augmented.dtype)])
        adjoint = augmented.T.conj()
        expected = xp.linalg.solve(adjoint @ augmented, adjoint @ augmented_rhs)

        result = LeastSquaresSolver("normal_solve").solve(problem, rhs)

        assert result.dtype == expected.dtype
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_normal_solve_numpy_rejects_an_exactly_singular_matrix():
    """Direct solves do not silently substitute a pseudoinverse."""
    with backend_context("numpy"):
        problem = LeastSquaresProblem(np.ones((3, 2)))
        with pytest.raises(np.linalg.LinAlgError, match="Singular"):
            LeastSquaresSolver("normal_solve").solve(problem, np.ones(3))


def test_constrained_map_declares_the_dtype_of_complex_constraints():
    """A complex constraint can produce complex values from real data."""
    operator = sparse_least_squares_map(np.eye(2), [[1.0, 1j]])
    expected = np.array([[0.5, -0.5j], [0.5j, 0.5]])
    assert np.issubdtype(operator.dtype, np.complexfloating)
    np.testing.assert_allclose(operator.to_matrix(), expected, atol=1e-14)
    np.testing.assert_allclose(operator.rmatvec(np.ones(2)), expected.T.conj() @ np.ones(2))


@pytest.mark.requires_jax
@pytest.mark.parametrize("solver_name", ["svd", "normal_solve", "normal_pinv"])
@pytest.mark.parametrize("custom_builder", [False, True])
def test_dense_solver_first_compiled_use_does_not_poison_caches(solver_name, custom_builder):
    """Compiled and interactive solves can share one problem in either order."""
    import jax
    import jax.numpy as jnp

    matrix = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, -2.0]])
    problem = LeastSquaresProblem(
        _with_normal_product(
            matrix,
            (lambda: jnp.asarray(matrix).T @ jnp.asarray(matrix)) if custom_builder else None,
        ),
    )
    solver = LeastSquaresSolver(solver_name)
    rhs = jnp.array([1.0, 2.0, -0.5])
    expected = np.linalg.lstsq(matrix, np.asarray(rhs), rcond=None)[0]
    with jax.checking_leaks():
        compiled = jax.jit(lambda values: solver.solve(problem, values))
        np.testing.assert_allclose(compiled(rhs), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(solver.solve(problem, 2 * rhs), 2 * expected, atol=1e-13)
    np.testing.assert_allclose(compiled(3 * rhs), 3 * expected, atol=1e-13)


@pytest.mark.requires_jax
def test_constrained_analysis_reuses_sparse_storage_after_first_compiled_call(monkeypatch):
    """The KKT solve reuses Kompe's sparse forward and adjoint maps."""
    import jax
    import jax.numpy as jnp

    matrix = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, -2.0]])
    constraint = np.ones((1, 2))
    weights = np.array([1.0, 0.5, 2.0])
    operator = sparse_least_squares_map(matrix, constraint, sqrt_weights=weights)
    # Eliminate the constraint x0 + x1 = 0 independently of the KKT system.
    direction = np.array([1.0, -1.0])
    column = matrix @ direction
    expected = direction[:, None] * (column * weights**2)[None, :]
    expected /= np.sum((weights * column) ** 2)
    rhs = jnp.array([[1.0, 0.5], [2.0, -0.5], [1.0, 0.25]])
    probe = jnp.array([[0.25, 1.0], [2.0, 0.5]])
    with jax.checking_leaks():
        np.testing.assert_allclose(jax.jit(operator.matmat)(rhs), expected @ rhs)

    asarray = jnp.asarray

    def reject_index_transfer(values, *args, **kwargs):
        if isinstance(values, np.ndarray) and values.shape == (np.count_nonzero(matrix), 2):
            pytest.fail("Analysis must reuse its sparse device storage.")
        return asarray(values, *args, **kwargs)

    monkeypatch.setattr(jnp, "asarray", reject_index_transfer)
    np.testing.assert_allclose(operator.matmat(rhs), expected @ rhs)
    np.testing.assert_allclose(operator.rmatmat(probe), expected.T @ probe)


def test_normal_pinv_solves_block_rhs():
    """Normal-equation pseudo-inverse supports reusable RHS maps."""
    A = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


def test_normal_pinv_uses_normal_equation_cutoff():
    """Normal pseudo-inverse applies cutoff after forming A* A."""
    A = np.diag([1.0, 1e-8])
    rhs = np.array([1.0, 1e-8])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    np.testing.assert_allclose(solution, np.array([1.0, 0.0]))


def test_normal_pinv_keeps_modes_above_normal_equation_cutoff():
    """Normal pseudo-inverse keeps modes above the A* A cutoff."""
    A = np.diag([1.0, 1e-6])
    rhs = np.array([1.0, 1e-6])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    np.testing.assert_allclose(solution, np.array([1.0, 1.0]))


def test_normal_pinv_does_not_use_direct_solve(monkeypatch):
    """Normal pseudo-inverse also used for full-rank systems."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method="normal_pinv", tolerance=1e-13)

    def fail_solve(*args, **kwargs):
        raise AssertionError("normal_pinv should apply a pseudo-inverse, not solve")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


def test_normal_pinv_response_solver_reuses_factorization(monkeypatch):
    """Reusable normal-pinv response solves cache dense factors."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs_first = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    rhs_second = np.array([[0.0, 4.0], [2.5, -1.0], [1.5, 3.0]])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method="normal_pinv", tolerance=1e-13)
    solve_response = solver.prepare(problem)

    def fail_dense_assembly():
        raise AssertionError("response solver should reuse cached dense factors")

    monkeypatch.setattr(problem, "system_matrix", fail_dense_assembly)

    A_H = A.T.conj()
    normal_pinv = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True)
    np.testing.assert_allclose(solve_response(rhs_first), normal_pinv @ (A_H @ rhs_first))
    np.testing.assert_allclose(solve_response(rhs_second), normal_pinv @ (A_H @ rhs_second))


@pytest.mark.parametrize("representation", ["sparse", "matrix_free"])
@pytest.mark.parametrize("constrained", [False, True])
def test_prepared_normal_fit_preserves_structured_adjoint(
    monkeypatch, representation, constrained
):
    """A supplied normal builder must not force a dense rectangular data map."""
    import scipy.sparse as sp
    from scipy.linalg import null_space

    xp = get_array_module()
    matrix = np.tile(np.diag(np.linspace(1, 2, 20)), (5, 1))
    if representation == "sparse":
        operator = as_linear_map(sp.csr_matrix(matrix), input_shape=(2, 10), output_shape=(5, 20))
    else:
        array = xp.asarray(matrix)
        operator = LinearMap(
            shape=matrix.shape,
            dtype=matrix.dtype,
            matvec=lambda x: array @ x,
            rmatvec=lambda y: array.T @ y,
            matmat=lambda x: array @ x,
            rmatmat=lambda y: array.T @ y,
            backend_operands=(array,),
            input_shape=(2, 10),
            output_shape=(5, 20),
        )
    weights = np.linspace(0.5, 1.5, 100)
    penalty = 0.2 * np.eye(20)
    constraints = np.ones((1, 20)) if constrained else None
    normal = matrix.T @ (weights[:, None] ** 2 * matrix)
    problem = LeastSquaresProblem(
        _with_normal_product(operator, lambda: xp.asarray(normal)),
        sqrt_weights=weights,
        regularization=penalty,
        constraints=constraints,
    )
    original = LinearMap.to_matrix

    def no_data_materialization(operator, **kwargs):
        assert 100 not in operator.shape, "The data map must keep its structured action."
        return original(operator, **kwargs)

    monkeypatch.setattr(LinearMap, "to_matrix", no_data_materialization)
    solve = LeastSquaresSolver("normal_pinv", tolerance=1e-12).prepare(problem)
    samples = xp.asarray(np.random.default_rng(312).normal(size=(5, 20, 2, 3)))
    Z = null_space(constraints) if constrained else np.eye(20)
    augmented = np.vstack([weights[:, None] * matrix, penalty]) @ Z
    rhs = np.vstack([weights[:, None] * np.asarray(samples).reshape(100, -1), np.zeros((20, 6))])
    expected = (Z @ np.linalg.lstsq(augmented, rhs, rcond=None)[0]).reshape(2, 10, 2, 3)
    np.testing.assert_allclose(solve(samples), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(solve(2 * samples), 2 * expected, rtol=1e-10, atol=1e-12)


def test_unregularized_system_reuses_data_operator():
    """Without regularization there is one canonical system map."""
    problem = LeastSquaresProblem(A=np.eye(3))

    assert problem.system_operator is problem.data_operator


def test_normal_pinv_discards_only_derived_regularized_matrix():
    """Keep the data matrix used by repeated solves, not its augmented copy."""
    problem = LeastSquaresProblem(
        A=np.eye(3),
        regularization=relative_regularization(np.eye(3), np.eye(3), 0.1),
    )
    regularized_system = problem.system_operator

    problem.dense_normal_pinv(1e-13)

    xp = get_array_module()
    assert regularized_system is not problem.data_operator
    assert xp not in regularized_system._dense_cache
    assert xp in problem.data_operator._dense_cache


def test_least_squares_requires_at_least_one_rhs_term():
    """A missing right-hand side is an input error, not an implicit zero solve."""
    problem = LeastSquaresProblem(A=np.eye(2))

    with pytest.raises(ValueError, match="At least one right-hand-side"):
        LeastSquaresSolver(method="normal_pinv").solve(problem, None)

    solve_response = LeastSquaresSolver(method="normal_pinv").prepare(problem)
    with pytest.raises(ValueError, match="At least one right-hand-side"):
        solve_response(None)


def test_least_squares_rejects_ambiguous_rhs_layout():
    """Equal element counts do not make an unrelated array shape meaningful."""
    problem = LeastSquaresProblem(as_linear_map(np.eye(6), output_shape=(2, 3)))

    with pytest.raises(ValueError, match="incompatible with data_shape"):
        LeastSquaresSolver(method="normal_pinv").solve(problem, np.ones((3, 4)))


def test_normal_pinv_response_solver_uses_explicit_data_adjoint():
    """Repeated dense response solves do not revisit structured callbacks."""
    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    adjoint_applications = 0

    def rmatmat(values):
        nonlocal adjoint_applications
        adjoint_applications += 1
        return matrix.T @ values

    operator = LinearMap(
        shape=matrix.shape,
        dtype=matrix.dtype,
        matvec=lambda values: matrix @ values,
        rmatvec=lambda values: matrix.T @ values,
        matmat=lambda values: matrix @ values,
        rmatmat=rmatmat,
        dense_array=lambda xp: xp.asarray(matrix),
    )
    operator.to_matrix()
    problem = LeastSquaresProblem(A=operator)
    solver = LeastSquaresSolver(method="normal_pinv", tolerance=1e-13)
    solve_response = solver.prepare(problem)
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])

    expected = np.linalg.pinv(matrix) @ rhs
    np.testing.assert_allclose(solve_response(rhs), expected)
    np.testing.assert_allclose(solve_response(2 * rhs), 2 * expected)
    assert adjoint_applications == 0


def test_normal_pinv_solve_reuses_cached_pseudo_inverse(monkeypatch):
    """Repeated dense normal-pinv solves reuse the cached n^3 factor."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs_first = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    rhs_second = np.array([[0.0, 4.0], [2.5, -1.0], [1.5, 3.0]])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method="normal_pinv", tolerance=1e-13)
    calls = 0
    original_pinv = np.linalg.pinv

    def counted_pinv(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_pinv(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", counted_pinv)

    first = solver.solve(problem, rhs_first)
    second = solver.solve(problem, rhs_second)

    A_H = A.T.conj()
    normal_pinv = original_pinv(A_H @ A, rtol=solver.tolerance, hermitian=True)
    np.testing.assert_allclose(first, normal_pinv @ (A_H @ rhs_first))
    np.testing.assert_allclose(second, normal_pinv @ (A_H @ rhs_second))
    assert len(problem._dense_normal_pinv_cache) == 1
    assert calls == (0 if jax_enabled() else 1)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_svd_solves_and_preconditioners_share_factorization(backend, monkeypatch):
    """One SVD serves repeated solves and both spectral preconditioners."""
    matrix = np.diag([3j, 0.0, 2.0])
    rhs = np.array([1.0 + 1j, 2.0 - 1j, 0.5j])
    expected = np.linalg.pinv(matrix, rtol=1e-12) @ rhs
    with backend_context(backend):
        xp = get_array_module()
        original_svd = xp.linalg.svd
        calls = []

        def svd(*args, **kwargs):
            calls.append(args[0])
            return original_svd(*args, **kwargs)

        monkeypatch.setattr(xp.linalg, "svd", svd)
        problem = LeastSquaresProblem(A=xp.asarray(matrix))
        solver = LeastSquaresSolver("svd", tolerance=1e-12)
        with np.errstate(divide="raise", invalid="raise"):
            for scale in (1.0, 2.0):
                result = solver.solve(problem, xp.asarray(scale * rhs))
                assert isinstance(result, xp.ndarray)
                np.testing.assert_allclose(result, scale * expected, atol=1e-12)
            for algorithm, power in (("lsmr", 1), ("cgls", 2)):
                preconditioner = LeastSquaresSolver(
                    algorithm, tolerance=1e-12, preconditioner="pinv"
                ).build_preconditioner(problem)
                block = preconditioner.matmat(xp.eye(3))
                assert isinstance(block, xp.ndarray)
                np.testing.assert_allclose(
                    block, np.diag(np.array([1 / 3, 0, 1 / 2]) ** power), atol=1e-12
                )
        assert len(calls) == 1


@pytest.mark.requires_jax
def test_svd_cache_keeps_separate_numpy_and_jax_factors(monkeypatch):
    """Changing execution backend never reuses factors on the wrong device."""
    import jax.numpy as jnp

    problem = LeastSquaresProblem(A=np.diag([1.0, 2.0]))
    calls = []
    for xp in (np, jnp):
        original_svd = xp.linalg.svd

        def svd(*args, _svd=original_svd, _xp=xp, **kwargs):
            calls.append(_xp)
            return _svd(*args, **kwargs)

        monkeypatch.setattr(xp.linalg, "svd", svd)

    for configured in ("jax", "numpy"):
        with backend_context(configured):
            for requested, xp in (("numpy", np), ("jax", jnp)):
                factors = problem.svd(backend=requested)
                assert all(isinstance(value, xp.ndarray) for value in factors)
                assert problem.svd(backend=requested) is factors
    assert calls == [np, jnp]


@pytest.mark.requires_jax
@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd"])
@pytest.mark.parametrize("jax_source", ["configured", "operator", "rhs", "regularizer"])
def test_dense_solve_stays_on_jax(solver_name, jax_source, monkeypatch):
    """Every dense solve honors JAX data, operators, and backend settings."""
    import jax.numpy as jnp

    import kompe.math.least_squares_problem as problem_module
    import kompe.math.least_squares_solver as solver_module

    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    expected = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    if jax_source == "regularizer":
        expected /= 1.5  # L = A, with relative strength 0.5.

    def reject_numpy(*_args, **_kwargs):
        pytest.fail("JAX solve must not transfer to or factorize with NumPy")

    monkeypatch.setattr(np.linalg, "svd", reject_numpy)
    monkeypatch.setattr(np.linalg, "solve", reject_numpy)
    monkeypatch.setattr(np.linalg, "pinv", reject_numpy)
    monkeypatch.setattr(problem_module, "to_numpy", reject_numpy)
    monkeypatch.setattr(solver_module, "to_numpy", reject_numpy)
    with backend_context("jax" if jax_source == "configured" else "numpy"):
        problem = LeastSquaresProblem(
            A=jnp.asarray(matrix) if jax_source == "operator" else matrix,
            regularization=relative_regularization(
                jnp.asarray(matrix) if jax_source == "operator" else matrix,
                jnp.asarray(matrix) if jax_source == "regularizer" else None,
                0.5 if jax_source == "regularizer" else None,
            ),
        )
        result = LeastSquaresSolver(solver_name, tolerance=1e-12).solve(
            problem, jnp.asarray(rhs) if jax_source == "rhs" else rhs
        )
    assert isinstance(result, jnp.ndarray)
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.requires_jax
def test_normal_response_reuses_prepared_factors_across_backends(monkeypatch):
    """Changing RHS backend reuses the prepared inverse, not another solve."""
    import jax.numpy as jnp

    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    expected = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    calls = []
    for xp in (np, jnp):
        original_pinv = xp.linalg.pinv

        def pinv(*args, _pinv=original_pinv, _xp=xp, **kwargs):
            calls.append(_xp)
            return _pinv(*args, **kwargs)

        monkeypatch.setattr(xp.linalg, "pinv", pinv)

    with backend_context("numpy"):
        problem = LeastSquaresProblem(A=matrix)
        response = LeastSquaresSolver("normal_pinv", tolerance=1e-12).prepare(problem)
        assert calls == [np]  # Preparation still warms the ordinary path.
        for xp in (np, jnp, jnp, np):
            result = response(xp.asarray(rhs))
            assert isinstance(result, xp.ndarray)
            np.testing.assert_allclose(result, expected, atol=1e-12)
    assert calls == [np]


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_prepared_response_retains_factors_after_problem_cache_eviction(backend, monkeypatch):
    """Prepared responses own their factors beyond the shared cache lifetime."""
    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([1.0, 3.0, 0.5])
    expected = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    with backend_context("numpy"):
        problem = LeastSquaresProblem(A=matrix)
        response = LeastSquaresSolver("normal_pinv").prepare(problem)

    problem._dense_normal_pinv_cache.clear()

    def reject_refactorization(*_args, **_kwargs):
        pytest.fail("A prepared response must retain its factorization")

    monkeypatch.setattr(problem, "dense_normal_pinv", reject_refactorization)
    with backend_context(backend):
        xp = get_array_module()
        result = response(xp.asarray(rhs))
        assert isinstance(result, xp.ndarray)
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.requires_jax
@pytest.mark.parametrize("custom_builder", [False, True])
def test_normal_matrices_and_factors_respect_explicit_backend(custom_builder):
    """Explicit NumPy/JAX requests also govern custom-builder results."""
    import jax.numpy as jnp

    matrix = np.array([[2.0 + 1j, 0.0], [0.0, 3.0 - 1j], [1.0, -1j]])
    data_normal = matrix.T.conj() @ matrix
    expected = data_normal + 0.1 * np.median(np.diag(data_normal).real) * np.eye(2)
    problem = LeastSquaresProblem(
        A=_with_normal_product(
            jnp.asarray(matrix), (lambda: data_normal) if custom_builder else None
        ),
        regularization=relative_regularization(jnp.asarray(matrix), jnp.eye(2), 0.1),
    )
    for configured in ("jax", "numpy"):
        with backend_context(configured):
            for requested, xp in (("numpy", np), ("jax", jnp)):
                built_normal = problem.dense_normal_matrix(backend=requested)
                inverse = problem.dense_normal_pinv(1e-12, backend=requested)
                assert isinstance(built_normal, xp.ndarray)
                assert isinstance(inverse, xp.ndarray)
                np.testing.assert_allclose(built_normal, expected, atol=1e-12)
                np.testing.assert_allclose(
                    inverse, np.linalg.pinv(expected, hermitian=True), atol=1e-12
                )
                assert problem.dense_normal_pinv(1e-12, backend=requested) is inverse
    np.testing.assert_array_equal(data_normal, matrix.T.conj() @ matrix)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_cached_inverse_backend_selection_does_not_build_normal_matrix(backend):
    """A persisted inverse is sufficient even with a custom regularized fit."""
    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([1.0, 3.0, 0.5])
    normal = matrix.T @ matrix
    normal += 0.1 * np.median(np.diag(normal)) * np.eye(2)
    inverse = np.linalg.pinv(normal, hermitian=True)

    class PrecomputedCache:
        def get_or_create(self, category, identity, builder):
            assert category == "least_squares_normal_pinv"
            assert identity["backend"] == ("numpy" if backend == "numpy" else "jax.numpy")
            return inverse

    def reject_build():
        pytest.fail("A cache hit must not build or balance the normal matrix")

    with backend_context("numpy"):
        xp = get_array_module(backend=backend)
        problem = LeastSquaresProblem(
            A=_with_normal_product(matrix, reject_build),
            regularization=relative_regularization(matrix, xp.eye(2), 0.1),
            operator_cache=PrecomputedCache(),
            cache_identity="precomputed",
        )
        solver = LeastSquaresSolver("normal_pinv")
        for solve in (
            lambda values: solver.solve(problem, values),
            solver.prepare(problem),
        ):
            result = solve(rhs)
            assert isinstance(result, xp.ndarray)
            np.testing.assert_allclose(result, inverse @ (matrix.T @ rhs), atol=1e-12)
        assert "system_operator" not in problem.__dict__


@pytest.mark.requires_jax
@pytest.mark.parametrize("jax_source", ["operator", "regularizer"])
def test_normal_response_prepares_only_the_operand_backend(jax_source):
    """Preparing a JAX-backed response must not also build CPU matrices."""
    import jax.numpy as jnp

    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    with backend_context("numpy"):
        problem = LeastSquaresProblem(
            A=jnp.asarray(matrix) if jax_source == "operator" else matrix,
            regularization=relative_regularization(
                jnp.asarray(matrix) if jax_source == "operator" else matrix,
                jnp.asarray(matrix) if jax_source == "regularizer" else None,
                0.5 if jax_source == "regularizer" else None,
            ),
        )
        response = LeastSquaresSolver("normal_pinv").prepare(problem)
        assert np not in problem.data_operator._dense_cache
        assert isinstance(problem.data_operator._dense_cache[jnp], jnp.ndarray)
        assert isinstance(response(np.ones(3)), jnp.ndarray)


@pytest.mark.requires_jax
def test_numpy_spectral_preconditioner_accepts_jax_operands():
    """Preconditioner application follows its operands after NumPy construction."""
    import jax.numpy as jnp

    with backend_context("numpy"):
        problem = LeastSquaresProblem(A=np.diag([2.0, 4.0]))
        preconditioner = LeastSquaresSolver("lsmr", preconditioner="pinv").build_preconditioner(
            problem
        )
        for method, operand in (
            ("matvec", jnp.ones(2)),
            ("rmatvec", jnp.ones(2)),
            ("matmat", jnp.eye(2)),
            ("rmatmat", jnp.eye(2)),
        ):
            result = getattr(preconditioner, method)(operand)
            assert isinstance(result, jnp.ndarray)
            np.testing.assert_allclose(result, np.diag([0.5, 0.25]) @ np.asarray(operand))


# Dense JAX solves


@pytest.mark.requires_jax
@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv"])
def test_dense_solvers_preserve_jax_output_when_backend_enabled(solver_name):
    """Dense solvers preserve JAX output when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_matrix = problem.system_matrix()
        assert "jax" in type(rhs_block).__module__
        assert "jax" in type(system_matrix).__module__
        solver = LeastSquaresSolver(method=solver_name, tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.requires_jax
def test_uncached_normal_pinv_stays_on_jax(monkeypatch):
    """An in-memory JAX factorization does not cross through NumPy."""
    import kompe.math.least_squares_problem as problem_module

    problem = LeastSquaresProblem(A=np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]]))
    previous_backend = jax_enabled()

    def reject_host_transfer(_array):
        raise AssertionError("uncached JAX pseudo-inverse crossed to NumPy")

    monkeypatch.setattr(problem_module, "to_numpy", reject_host_transfer)
    try:
        set_backend("jax")
        normal_pinv = problem.dense_normal_pinv(1e-13)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(normal_pinv).__module__


@pytest.mark.requires_jax
def test_svd_solver_preserves_jax_output_when_backend_enabled():
    """SVD solver keeps JAX-facing assembly and output."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_matrix = problem.system_matrix()
        assert "jax" in type(rhs_block).__module__
        assert "jax" in type(system_matrix).__module__
        solver = LeastSquaresSolver(method="svd", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.requires_jax
def test_least_squares_problem_follows_jax_operator_context_when_numpy_active():
    """JAX-backed operator terms should drive matrix-free assembly."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])

    try:
        set_backend("numpy")
        problem = LeastSquaresProblem(A=jnp.asarray(A))
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_block = problem.system_operator.matmat(np.eye(2))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(rhs_block).__module__
    assert "jax" in type(system_block).__module__
    np.testing.assert_allclose(np.asarray(system_block), A)


@pytest.mark.requires_jax
def test_normal_pinv_matches_numpy_hermitian_reference_when_jax_enabled():
    """JAX normal-pinv matches the hermitian reference."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(method="normal_pinv", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


# Iterative solvers


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_solver_solves_block_rhs_with_base_preconditioner(solver_name):
    """Iterative block RHS solves reuse the base preconditioner."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0, -1.0], [3.0, 1.0, 0.5], [0.5, -2.0, 4.0], [1.5, 0.0, 2.0]])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method=solver_name, tolerance=1e-12, preconditioner="jacobi")
    preconditioner = solver.build_preconditioner(problem)

    assert preconditioner.shape == (2, 2)
    solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_solvers_do_not_materialize_dense_system(monkeypatch, solver_name):
    """Iterative solves stay matrix-free for no dense preconditioner."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method=solver_name, tolerance=1e-12)

    def fail_dense_assembly():
        raise AssertionError("iterative solvers should not assemble dense systems")

    monkeypatch.setattr(problem, "system_matrix", fail_dense_assembly)

    solution = solver.solve(problem, rhs, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_jacobi_preconditioner_does_not_materialize_dense_system(
    monkeypatch, solver_name
):
    """Jacobi-preconditioned iterative solves stay matrix-free."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A)
    solver = LeastSquaresSolver(method=solver_name, tolerance=1e-12, preconditioner="jacobi")

    def fail_dense_assembly():
        raise AssertionError(
            "jacobi-preconditioned iterative solvers should not assemble dense systems"
        )

    monkeypatch.setattr(problem, "system_matrix", fail_dense_assembly)

    preconditioner = solver.build_preconditioner(problem)
    solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.requires_jax
@pytest.mark.parametrize("solver_name", ["cgls", "lsmr"])
@pytest.mark.parametrize("preconditioner_type", [None, "jacobi", "pinv"])
def test_iterative_solvers_preserve_jax_output_when_backend_enabled(
    solver_name, preconditioner_type
):
    """Iterative solvers preserve JAX output when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(
            method=solver_name, tolerance=1e-12, preconditioner=preconditioner_type
        )
        preconditioner = solver.build_preconditioner(problem)
        solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.requires_jax
def test_jax_lsmr_solves_underdetermined_block_rhs():
    """Internal JAX LSMR handles rectangular underdetermined systems."""
    A = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(method="lsmr", tolerance=1e-12)
        solution = solver.solve(problem, rhs, maxiter=200)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.requires_jax
@pytest.mark.parametrize("complex_system", [False, True])
def test_jax_lsmr_recurrence_matches_scipy(complex_system):
    """Internal LSMR matches SciPy for damping and an initial guess."""
    import jax.numpy as jnp

    from kompe.math.jax_iterative import lsmr as jax_lsmr

    rng = np.random.default_rng(2841)
    matrix = rng.normal(size=(8, 5))
    rhs = rng.normal(size=8)
    initial_guess = rng.normal(size=5)
    if complex_system:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        rhs = rhs + 1j * rng.normal(size=rhs.shape)
        initial_guess = initial_guess + 1j * rng.normal(size=initial_guess.shape)

    options = {
        "damp": 0.25,
        "atol": 1e-12,
        "btol": 1e-12,
        "conlim": 1e10,
        "maxiter": 100,
        "x0": initial_guess,
    }
    expected = scipy_lsmr(matrix, rhs, **options)
    actual = jax_lsmr(as_linear_map(jnp.asarray(matrix)), jnp.asarray(rhs), **options)

    assert actual[1:3] == expected[1:3]
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual[3:], expected[3:], rtol=1e-11, atol=1e-12)


@pytest.mark.requires_jax
def test_jax_lsmr_zero_rhs_discards_non_solution_initial_guess():
    """A zero RHS rejects an initial guess that is not a solution."""
    import jax.numpy as jnp

    from kompe.math.jax_iterative import lsmr as jax_lsmr

    matrix = np.array([[2.0, -1.0], [1.0, 3.0], [0.5, 2.0]])
    rhs = np.zeros(3)
    initial_guess = np.array([1.0, -2.0])

    expected = scipy_lsmr(matrix, rhs, x0=initial_guess)
    actual = jax_lsmr(as_linear_map(jnp.asarray(matrix)), jnp.asarray(rhs), x0=initial_guess)

    assert actual[1:3] == expected[1:3]
    np.testing.assert_allclose(actual[0], expected[0])


@pytest.mark.requires_jax
def test_jax_lsmr_uses_complex_operator_dtype():
    """A complex operator promotes real right-hand sides correctly."""
    import jax.numpy as jnp

    from kompe.math.jax_iterative import lsmr as jax_lsmr

    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    rhs = rng.normal(size=8)

    solution, stop_code, *_ = jax_lsmr(
        as_linear_map(jnp.asarray(matrix)), jnp.asarray(rhs), atol=1e-12, btol=1e-12, maxiter=100
    )

    assert int(stop_code) in {1, 2}
    assert np.issubdtype(solution.dtype, np.complexfloating)
    expected = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


# Public solver validation


@pytest.mark.parametrize("weight", [-1.0, np.inf, np.nan, np.array([1.0, 2.0])])
def test_relative_regularization_strength_must_be_finite_non_negative_scalar(weight):
    """Invalid regularization weights fail before system assembly."""
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        LeastSquaresProblem(
            A=np.eye(2),
            regularization=relative_regularization(np.eye(2), np.eye(2), weight),
        )


@pytest.mark.parametrize("algorithm", LeastSquaresSolver.VALID_SOLVERS)
def test_exact_constraints_preserve_dtype_with_a_custom_normal_builder(algorithm):
    """Complex independent coordinates preserve the original objective."""
    from kompe.math import null_space_linear_map

    data = np.array([[2.0, 1.0, 0.0]])
    data_normal = data.T @ data
    constraints = np.array([[1.0, 1j, 0.0], [0.0, 0.0, 1.0]])
    basis = null_space_linear_map(constraints)
    problem = LeastSquaresProblem(_with_normal_product(data, lambda: data_normal))
    restricted = problem.restrict_solution(basis)
    coordinates = LeastSquaresSolver(algorithm, tolerance=1e-12).solve(restricted, np.array([3.0]))
    result = basis(coordinates)
    np.testing.assert_allclose(data @ result, [3.0], atol=1e-12)
    np.testing.assert_allclose(constraints @ result, 0.0, atol=1e-12)
    np.testing.assert_array_equal(data_normal, data.T @ data)
    assert np.issubdtype(result.dtype, np.complexfloating)


def test_regularization_does_not_mutate_a_custom_data_normal_matrix():
    """A supplied normal matrix remains owned by its builder."""
    data_normal = np.eye(2)
    problem = LeastSquaresProblem(
        A=_with_normal_product(np.eye(2), lambda: data_normal),
        regularization=relative_regularization(np.eye(2), np.eye(2), 1.0),
    )

    np.testing.assert_allclose(problem.dense_normal_matrix(), 2.0 * np.eye(2))
    np.testing.assert_array_equal(data_normal, np.eye(2))


@pytest.mark.parametrize("dtype", ["int64", "float32", "complex64"])
@pytest.mark.parametrize("kind", ["diagonal", "dense", "matrix_free"])
def test_custom_normal_regularization_promotes_integer_arithmetic(dtype, kind):
    """Fractional penalties survive integer input without widening inexact arrays."""
    xp = get_array_module()
    data = xp.asarray([[1, 2], [0, 1]], dtype=dtype)
    diagonal = xp.asarray([1, 2], dtype=dtype)
    penalty = xp.diag(diagonal)
    regularizer = diagonal if kind == "diagonal" else penalty
    if kind == "matrix_free":
        regularizer = LinearMap(
            shape=penalty.shape,
            dtype=penalty.dtype,
            matvec=lambda x: penalty @ x,
            rmatvec=lambda y: penalty.T.conj() @ y,
            matmat=lambda x: penalty @ x,
            rmatmat=lambda y: penalty.T.conj() @ y,
            normal_matrix_diag=lambda row_scale=None: (
                np.array([1.0, 4.0]) * (1 if row_scale is None else np.abs(row_scale) ** 2)
            ),
            backend_operands=(penalty,),
        )
    data_normal = data.T.conj() @ data
    problem = LeastSquaresProblem(
        _with_normal_product(data, lambda: data_normal),
        regularization=relative_regularization(data, regularizer, 0.25),
    )
    # The data and penalty normal diagonals have medians 3 and 2.5.
    scale = np.sqrt(0.25 * 3.0 / 2.5)
    augmented = np.vstack([np.array([[1.0, 2.0], [0.0, 1.0]]), scale * np.diag([1.0, 2.0])])
    expected = augmented.T @ augmented

    normal = problem.dense_normal_matrix()

    assert normal.dtype == xp.result_type(data.dtype, 0.0)
    np.testing.assert_allclose(normal, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(data_normal, [[1, 2], [2, 5]])
    rhs = xp.asarray([2.0, -1.0])
    actual = LeastSquaresSolver("normal_solve").solve(problem, rhs)
    reference = np.linalg.lstsq(augmented, [2.0, -1.0, 0.0, 0.0], rcond=None)[0]
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("materialized", [False, True])
def test_penalty_normal_uses_bounded_blocks_or_an_existing_matrix(
    complex_values, materialized, monkeypatch
):
    """A tall penalty needs no full temporary matrix; dense penalties are reused."""
    import kompe.math.linear_map as problem_module

    rng = np.random.default_rng(405)
    data = rng.normal(size=(17, 11))
    penalty = rng.normal(size=(23, 11))
    if complex_values:
        data = data + 1j * rng.normal(size=data.shape)
        penalty = penalty + 1j * rng.normal(size=penalty.shape)
    xp = get_array_module()
    matrix = xp.asarray(penalty)
    calls = []

    def matmat(columns):
        calls.append(columns.shape[1])
        assert columns.shape[1] <= 3
        return matrix @ columns

    regularizer = LinearMap(
        shape=matrix.shape,
        dtype=matrix.dtype,
        matvec=lambda x: matrix @ x,
        rmatvec=lambda y: matrix.T.conj() @ y,
        matmat=matmat,
        rmatmat=lambda y: matrix.T.conj() @ y,
        dense_array=lambda xp: xp.asarray(penalty),
        normal_matrix_diag=lambda row_scale=None: np.sum(
            np.abs(penalty) ** 2
            * (1 if row_scale is None else np.abs(np.asarray(row_scale))[:, None] ** 2),
            axis=0,
        ),
        backend_operands=(matrix,),
    )
    if materialized:
        regularizer.to_matrix()
    # Account for the input, penalty output, and adjoint output blocks.
    bytes_per_column = (2 * data.shape[1] + penalty.shape[0]) * matrix.dtype.itemsize
    monkeypatch.setattr(problem_module, "_NORMAL_MATRIX_WORK_BYTES", 3 * bytes_per_column)
    data_normal = data.T.conj() @ data
    problem = LeastSquaresProblem(
        _with_normal_product(xp.asarray(data), lambda: xp.asarray(data_normal)),
        regularization=relative_regularization(xp.asarray(data), regularizer, 0.2),
    )
    scale = np.sqrt(
        0.2
        * np.median(np.sum(np.abs(data) ** 2, axis=0))
        / np.median(np.sum(np.abs(penalty) ** 2, axis=0))
    )
    augmented = np.vstack([data, scale * penalty])

    normal = problem.dense_normal_matrix()

    assert isinstance(normal, xp.ndarray)
    np.testing.assert_allclose(normal, augmented.T.conj() @ augmented, rtol=1e-12, atol=1e-12)
    assert calls == ([] if materialized else [3, 3, 3, 2])
    assert (regularizer.materialized_matrix is not None) == materialized
    if jax_enabled():
        import jax

        with jax.checking_leaks():
            compiled = jax.jit(problem.dense_normal_matrix)()
        np.testing.assert_allclose(compiled, normal, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("algorithm", ["normal_solve", "normal_pinv"])
@pytest.mark.parametrize("diagonal_penalty", [False, True])
def test_native_normal_builder_preserves_a_weighted_constrained_objective(
    algorithm, diagonal_penalty, monkeypatch
):
    """Native arrays preserve weights, complex gauges, and regularization scales."""
    from scipy.linalg import null_space

    import kompe.math.least_squares_problem as problem_module
    from kompe.math import null_space_linear_map

    rng = np.random.default_rng(404)
    data = rng.normal(size=(8, 5)) + 1j * rng.normal(size=(8, 5))
    weights = rng.uniform(0.4, 2.0, size=8)
    penalty = np.diag(np.arange(1.0, 6.0)) if diagonal_penalty else rng.normal(size=(3, 5))
    constraints = np.array([[1.0, 1j, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]])
    rhs = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
    weighted = weights[:, None] * data
    scale = np.sqrt(
        0.2
        * np.median(np.sum(np.abs(weighted) ** 2, axis=0))
        / np.median(np.sum(np.abs(penalty) ** 2, axis=0))
    )
    reference_basis = null_space(constraints)
    augmented = np.vstack([weighted, scale * penalty]) @ reference_basis
    target = np.vstack([weights[:, None] * rhs, np.zeros((penalty.shape[0], 2))])
    expected = reference_basis @ np.linalg.lstsq(augmented, target, rcond=None)[0]
    xp = get_array_module()
    data_normal = xp.asarray(weighted.T.conj() @ weighted)
    problem = LeastSquaresProblem(
        _with_normal_product(xp.asarray(data), lambda: data_normal),
        sqrt_weights=xp.asarray(weights),
        regularization=relative_regularization(
            xp.asarray(data),
            xp.asarray(np.diag(penalty) if diagonal_penalty else penalty),
            0.2,
            sqrt_weights=xp.asarray(weights),
        ),
    )
    basis = null_space_linear_map(constraints)
    transfer = problem_module.to_numpy
    transfers = []

    def record(values):
        transfers.append(values.shape)
        return transfer(values)

    monkeypatch.setattr(problem_module, "to_numpy", record)
    restricted = problem.restrict_solution(basis)
    actual = basis(
        LeastSquaresSolver(algorithm, tolerance=1e-12).solve(restricted, xp.asarray(rhs))
    )
    assert isinstance(actual, xp.ndarray)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(constraints @ actual, 0.0, atol=1e-12)
    np.testing.assert_allclose(data_normal, weighted.T.conj() @ weighted, atol=1e-12)
    assert basis.materialized_matrix is None
    if jax_enabled():
        assert all(len(shape) <= 1 for shape in transfers)


@pytest.mark.parametrize("algorithm", ["lsmr", "cgls"])
def test_iterative_regularized_fit_does_not_call_a_normal_builder(algorithm, monkeypatch):
    """Scaling and exact constraints do not force matrix-free fits to become dense."""
    from kompe.math import null_space_linear_map

    matrix = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
    xp = get_array_module()
    data = xp.asarray(matrix)

    def unexpected(*args, **kwargs):
        pytest.fail("Iterative fitting needs no explicit matrix or normal builder.")

    operator = LinearMap(
        shape=data.shape,
        dtype=data.dtype,
        matvec=lambda x: data @ x,
        rmatvec=lambda y: data.T @ y,
        matmat=lambda x: data @ x,
        normal_matrix_diag=lambda row_scale=None: np.sum(
            matrix**2 * (1 if row_scale is None else np.abs(np.asarray(row_scale))[:, None] ** 2),
            axis=0,
        ),
        dense_array=unexpected,
        backend_operands=(data,),
    )
    problem = LeastSquaresProblem(
        _with_normal_product(operator, unexpected),
        regularization=relative_regularization(operator, np.ones(3), 0.1),
    )
    basis = null_space_linear_map([[0.0, 0.0, 1.0]])
    restricted = problem.restrict_solution(basis)
    to_matrix = LinearMap.to_matrix

    def existing_matrix_only(operator, **kwargs):
        # Compact QR reflectors already exist as arrays; their backend
        # copies are not materializations of the nullspace or data map.
        if operator.materialized_matrix is None:
            unexpected()
        return to_matrix(operator, **kwargs)

    monkeypatch.setattr(LinearMap, "to_matrix", existing_matrix_only)
    rhs = np.array([1.0, 2.0, 3.0, 4.0])
    actual = basis(
        LeastSquaresSolver(algorithm, tolerance=1e-12).solve(restricted, xp.asarray(rhs))
    )
    scale = np.sqrt(0.1 * np.median(np.sum(matrix**2, axis=0)))
    augmented = np.vstack([matrix[:, :2], scale * np.eye(2)])
    expected = np.r_[np.linalg.lstsq(augmented, np.r_[rhs, 0.0, 0.0], rcond=None)[0], 0.0]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    assert operator.materialized_matrix is None


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_solver_tolerance_must_be_a_finite_non_negative_scalar(tolerance):
    """Reject solver tolerances that cannot define a numerical cutoff."""
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        LeastSquaresSolver(tolerance=tolerance)


def test_solver_tolerance_must_be_scalar_numeric_data():
    """Reject booleans explicitly and let array conversion report its own error."""
    with pytest.raises(TypeError, match="finite non-negative scalar"):
        LeastSquaresSolver(tolerance=True)
    with pytest.raises(TypeError):
        LeastSquaresSolver(tolerance=np.array([1.0, 2.0]))


@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd"])
@pytest.mark.parametrize("entrypoint", ["solve", "prepare"])
def test_dense_solvers_reject_explicit_preconditioners(solver_name, entrypoint):
    """Dense solvers reject explicitly supplied preconditioners."""
    problem = LeastSquaresProblem(A=np.eye(2))
    solver = LeastSquaresSolver(method=solver_name)
    preconditioner = as_linear_map(np.eye(2))

    with pytest.raises(ValueError, match="does not accept a preconditioner"):
        if entrypoint == "solve":
            solver.solve(problem, np.ones(2), preconditioner=preconditioner)
        else:
            solver.prepare(problem, preconditioner=preconditioner)


@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd"])
@pytest.mark.parametrize("option, value", [("damp", 1.0), ("maxiter", 20)])
def test_dense_solvers_reject_unsupported_options(solver_name, option, value):
    """Changing algorithms must not silently discard solver options."""
    problem = LeastSquaresProblem(A=np.eye(2))

    with pytest.raises(TypeError, match=f"unexpected keyword argument '{option}'"):
        LeastSquaresSolver(solver_name).solve(problem, np.ones(2), **{option: value})


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_lsmr_damping_does_not_change_meaning_with_preconditioning(backend):
    """Damping cannot silently move its penalty to transformed coordinates."""
    with backend_context(backend):
        xp = get_array_module()
        identity = xp.eye(2)
        rhs = xp.ones(2)
        problem = LeastSquaresProblem(A=identity)
        preconditioner = as_linear_map(xp.asarray([2.0, 3.0]))
        solver = LeastSquaresSolver("lsmr", tolerance=1e-12)

        with pytest.raises(ValueError, match="damp.*right preconditioner"):
            solver.solve(problem, rhs, damp=1.0, preconditioner=preconditioner)

        np.testing.assert_allclose(
            solver.solve(problem, rhs, damp=0.0, preconditioner=preconditioner), rhs
        )
        np.testing.assert_allclose(solver.solve(problem, rhs, damp=1.0), 0.5 * rhs)

        # An explicit penalty stays in the original coefficient coordinates.
        regularized = LeastSquaresProblem(A=[identity, identity])
        np.testing.assert_allclose(
            solver.solve(regularized, [rhs, None], preconditioner=preconditioner), 0.5 * rhs
        )


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
@pytest.mark.parametrize("complex_system", [False, True])
@pytest.mark.parametrize("materialized", [False, True])
def test_lsmr_normalization_reuses_dense_data_for_each_rhs(
    scale, complex_system, materialized, monkeypatch
):
    """RHS-specific scaling preserves damping without copying an m-by-n matrix."""
    import kompe.math.linear_map as module

    rng = np.random.default_rng(361)
    matrix = rng.normal(size=(7, 4))
    rhs = rng.normal(size=(7, 3))
    if complex_system:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        rhs = rhs + 1j * rng.normal(size=rhs.shape)
    expected = np.linalg.lstsq(
        np.vstack([matrix, 0.25 * np.eye(4)]), np.vstack([rhs, np.zeros((4, 3))]), rcond=None
    )[0]
    xp = get_array_module()
    data = xp.asarray(scale * matrix)
    problem = LeastSquaresProblem(data)
    operator = problem.system_operator
    if materialized:
        operator.to_matrix()
    as_map = module.as_linear_map

    def reject_matrix_copy(value, *args, **kwargs):
        if getattr(value, "shape", None) == data.shape and not isinstance(value, LinearMap):
            raise AssertionError("LSMR must scale operator actions, not create a matrix copy.")
        return as_map(value, *args, **kwargs)

    monkeypatch.setattr(module, "as_linear_map", reject_matrix_copy)
    actual = LeastSquaresSolver("lsmr", tolerance=1e-12).solve(
        problem, xp.asarray(scale * rhs), damp=scale * 0.25
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(data, scale * matrix)
    assert bool(operator._dense_cache) == materialized


@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd"])
def test_dense_solvers_do_not_build_configured_preconditioners(solver_name):
    """A requested preconditioner is never silently ignored."""
    problem = LeastSquaresProblem(A=np.eye(2))
    solver = LeastSquaresSolver(method=solver_name, preconditioner="jacobi")

    with pytest.raises(ValueError, match="does not accept a preconditioner"):
        solver.build_preconditioner(problem)
    with pytest.raises(ValueError, match="does not accept a preconditioner"):
        solver.solve(problem, np.ones(2))


@pytest.mark.parametrize("algorithm", ["lsmr", "cgls"])
@pytest.mark.parametrize("kind", ["jacobi", "pinv"])
def test_configured_preconditioners_are_applied_and_reused(algorithm, kind, monkeypatch):
    """Shared solver settings apply without a manual build-and-pass step."""
    problem = LeastSquaresProblem(np.diag([1.0, 100.0, 1e4]))
    solver = LeastSquaresSolver(algorithm, tolerance=1e-12, preconditioner=kind)
    preconditioner = solver.build_preconditioner(problem)

    def unexpected(*args, **kwargs):
        raise AssertionError("The problem's preconditioner should be reused.")

    monkeypatch.setattr(solver, "_build_jacobi_preconditioner", unexpected)
    monkeypatch.setattr(solver, "_build_pinv_preconditioner", unexpected)
    rhs = np.array([1.0, 100.0, 1e4])
    for _ in range(2):
        actual = solver.solve(problem, rhs, maxiter=2)
        np.testing.assert_allclose(actual, 1.0, atol=1e-10)
    np.testing.assert_allclose(solver.prepare(problem)(rhs), 1.0, atol=1e-10)
    assert solver.build_preconditioner(problem) is preconditioner
