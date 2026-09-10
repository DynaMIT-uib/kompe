"""Literal inverse-problem objectives and explicit relative penalties."""

import numpy as np
import pytest

from kompe.math import (
    LeastSquaresProblem,
    LeastSquaresSolver,
    LinearMap,
    as_linear_map,
    get_array_module,
    relative_regularization,
)


@pytest.mark.parametrize("method", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("data_scale", [0.0, 1.0, 7.0])
def test_literal_penalties_do_not_depend_on_data_scale(method, data_scale):
    xp = get_array_module()
    A = xp.asarray([[2.0, 1.0], [0.0, 3.0], [1.0, -0.5]]) * data_scale
    weights = xp.asarray([1.0, 0.5, 2.0])
    penalties = [xp.asarray([0.3, 4.0]), xp.asarray([[1.0, -2.0]])]
    rhs = xp.asarray([1.0, 2.0, -1.0])
    problem = LeastSquaresProblem(A, sqrt_weights=weights, regularization=penalties)
    stacked = np.vstack([np.asarray(weights[:, None] * A), np.diag(penalties[0]), penalties[1]])
    expected = np.linalg.lstsq(stacked, np.r_[weights * rhs, np.zeros(3)], rcond=None)[0]

    actual = LeastSquaresSolver(method, tolerance=1e-12).solve(problem, rhs)

    assert isinstance(actual, xp.ndarray)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(problem.regularization_operators[0].diagonal(), penalties[0])
    np.testing.assert_allclose(problem.system_matrix(), stacked)


def test_literal_matrix_free_penalty_needs_no_scale_probe(monkeypatch):
    xp = get_array_module()
    penalty = xp.asarray([[1.0, -2.0]])
    operator = LinearMap(
        shape=(1, 2),
        dtype=penalty.dtype,
        matvec=lambda x: penalty @ x,
        rmatvec=lambda y: penalty.T @ y,
        matmat=lambda x: penalty @ x,
        rmatmat=lambda y: penalty.T @ y,
        backend_operands=(penalty,),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("A supplied objective must not infer a penalty scale.")

    monkeypatch.setattr(LinearMap, "normal_matrix_diag", unexpected)
    problem = LeastSquaresProblem(xp.eye(2), regularization=operator)
    actual = LeastSquaresSolver("normal_solve").solve(problem, xp.asarray([1.0, 2.0]))
    expected = np.linalg.solve(np.eye(2) + np.asarray(penalty.T @ penalty), [1.0, 2.0])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_relative_penalty_retains_the_documented_weighted_objective():
    xp = get_array_module()
    A = xp.asarray([[1.0, 3.0], [2.0, -1.0], [0.0, 2.0]])
    weights = xp.asarray([0.5, 2.0, 1.0])
    L = as_linear_map(xp.asarray([1.0, 4.0]))
    R = relative_regularization(A, L, 0.2, sqrt_weights=weights)
    scale = np.sqrt(
        0.2
        * np.median(np.sum(np.asarray(weights[:, None] * A) ** 2, axis=0))
        / np.median([1.0, 16.0])
    )
    assert R.is_diagonal
    assert not R._dense_cache
    np.testing.assert_allclose(R.diagonal(), scale * np.asarray([1.0, 4.0]))


def test_disabled_relative_penalty_does_not_inspect_operators(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled relative regularization must do no numerical work.")

    monkeypatch.setattr(LinearMap, "normal_matrix_diag", unexpected)
    assert relative_regularization(None, None, 0.0) is None
    assert relative_regularization(None, None, None) is None
