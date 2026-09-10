"""Mathematical contracts for sampled constant linear evolution."""

import numpy as np
import pytest

from kompe.math import (
    LINEAR_EVOLUTION_METHODS,
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
    get_array_module,
    linear_evolution,
    prepare_linear_evolution,
)


@pytest.mark.parametrize(
    "integrator", ["euler", "exponential", "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
)
def test_sampling_does_not_change_the_integrators_accepted_trajectory(integrator):
    """Dense and final-only output solve the same IVP."""
    xp = get_array_module()
    A = diagonal_linear_map(xp.array([-1.0]))
    stepper = prepare_linear_evolution(
        A,
        xp.ones(1),
        method=integrator,
        dt=0.1 if integrator == "euler" else None,
        rtol=1e-10,
        atol=1e-12,
    )
    times = np.array([0, 0.03, 0.12, 0.29, 0.5, 0.71, 0.735])
    sampled = np.concatenate(list(stepper(xp.array([2.0]), times)), axis=-1)[0]
    final = next(stepper(xp.array([2.0]), [times[-1]]))
    if integrator == "euler":
        steps = np.floor(times / 0.1).astype(int)
        expected = 1 + 0.9**steps * (1 - (times - 0.1 * steps))
    else:
        expected = 1 + np.exp(-times)
    np.testing.assert_allclose(sampled, expected, rtol=1e-9)
    np.testing.assert_allclose(sampled[-1], final[0], rtol=1e-13)


@pytest.mark.parametrize("integrator", ["euler", "exponential", "RK45", "Radau", "BDF", "LSODA"])
def test_forced_null_mode_evolves_without_an_equilibrium(integrator):
    """Integrate x' = 1 despite the lack of a stationary state."""
    xp = get_array_module()
    A = diagonal_linear_map(xp.zeros(1))
    advance = prepare_linear_evolution(
        A, xp.ones(1), method=integrator, dt=0.1 if integrator == "euler" else None
    )
    actual = next(advance(xp.array([2.0]), [0.735]))[..., 0]
    np.testing.assert_allclose(actual, [2.735], atol=1e-13)


def test_exponential_reuses_fixed_duration_maps_without_equilibrium(monkeypatch):
    """Bound and reuse propagators independently of equilibrium."""
    calls = []
    original = linear_evolution.affine_exponential

    def counted(A, b, duration):
        calls.append(duration)
        return original(A, b, duration)

    monkeypatch.setattr(linear_evolution, "affine_exponential", counted)
    xp = get_array_module()
    advance = prepare_linear_evolution(diagonal_linear_map(-xp.ones(1)), xp.ones(1))
    for duration in [0.1, 0.2, 0.3, 0.4, 0.1, 0.5, 0.1, 0.2]:
        next(advance(xp.ones(1), [duration]))
    np.testing.assert_allclose(calls, [0.1, 0.2, 0.3, 0.4, 0.5, 0.2])


def test_scipy_adapts_once_per_input_interval_not_per_output(monkeypatch):
    """Keep adaptive work independent of output frequency."""
    original = linear_evolution.scipy.integrate.RK45
    runs = []

    class CountedRK45(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            runs.append(self)

    monkeypatch.setattr(linear_evolution.scipy.integrate, "RK45", CountedRK45)
    advance = prepare_linear_evolution(
        as_linear_map(np.array([[-1.0]])), np.ones(1), method="RK45"
    )
    list(advance(np.array([2.0]), np.linspace(0, 1, 101)))
    first = runs[-1]
    list(advance(np.array([2.0]), [1.0]))
    assert len(runs) == 2
    assert first.nfev == runs[-1].nfev < 100


def test_uniform_exponential_outputs_share_one_propagator(monkeypatch):
    """Clock-subtraction roundoff must not rebuild a regular cadence."""
    durations = []
    original = linear_evolution.affine_exponential

    def counted(A, b, duration):
        durations.append(duration)
        return original(A, b, duration)

    monkeypatch.setattr(linear_evolution, "affine_exponential", counted)
    advance = prepare_linear_evolution(
        diagonal_linear_map(-get_array_module().ones(1)), get_array_module().ones(1)
    )
    times = np.arange(1, 30) * 0.1
    result = list(advance(np.array([2.0]), times, output_interval=0.1))
    assert durations == [0.1]
    np.testing.assert_allclose(result[-1][..., -1], 1 + np.exp(-times[-1]), atol=1e-14)


@pytest.mark.parametrize("method", LINEAR_EVOLUTION_METHODS)
@pytest.mark.parametrize("kind", ["matrix-free", "materialized", "diagonal"])
def test_scientific_shapes_and_structured_operators(method, kind):
    """Keep state axes and the operator's useful execution form."""
    xp = get_array_module()
    shape = (2, 2)
    A = LinearMap(
        shape=(4, 4),
        input_shape=shape,
        output_shape=shape,
        dtype=float,
        matvec=lambda x: -x,
        rmatvec=lambda x: -x,
    )
    if kind == "materialized":
        A.to_matrix()
    elif kind == "diagonal":
        A = diagonal_linear_map(-xp.ones(shape), input_shape=shape, output_shape=shape)
    initial = xp.arange(4.0).reshape(shape)
    b = xp.ones(shape)
    advance = prepare_linear_evolution(
        A,
        b,
        method=method,
        dt=0.1 if method == "euler" else None,
        rtol=1e-10,
        atol=xp.full(shape, 1e-12),
    )
    actual = next(advance(initial, [0.735]))[..., 0]
    decay = 0.9**7 * 0.965 if method == "euler" else np.exp(-0.735)
    assert actual.shape == shape
    np.testing.assert_allclose(actual, 1 + (initial - 1) * decay, rtol=1e-9)
    np.testing.assert_array_equal(initial, np.arange(4.0).reshape(shape))
    if method == "euler" and kind == "matrix-free":
        assert not A._dense_cache
    if kind == "diagonal" and method in ("euler", "exponential"):
        assert not A._dense_cache


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "typo"},
        {"method": "euler"},
        {"dt": 0.1},
        {"method": "euler", "dt": -1},
        {"method": "euler", "dt": True},
        {"method": "RK45", "rtol": 0},
        {"method": "RK45", "atol": [-1]},
        {"method": "RK45", "atol": [1, 2]},
        {"method": "RK45", "atol": True},
    ],
)
def test_invalid_equation_controls(kwargs):
    """Reject controls that cannot describe the requested method."""
    with pytest.raises(ValueError):
        prepare_linear_evolution(np.eye(1), np.ones(1), **kwargs)


@pytest.mark.parametrize("times", [[], [-1], [1, 0], [1, 1], [np.nan], [[1]]])
def test_invalid_sample_times(times):
    """Require ordered physical offsets."""
    advance = prepare_linear_evolution(np.eye(1), np.ones(1))
    with pytest.raises(ValueError):
        list(advance(np.ones(1), times))


def test_invalid_state_shapes():
    """Do not guess scientific axes from a flat element count."""
    with pytest.raises(ValueError, match="matching input and output"):
        prepare_linear_evolution(np.ones((2, 1)), np.ones(1))
    with pytest.raises(ValueError, match="b must"):
        prepare_linear_evolution(np.eye(2), np.ones((2, 1)))
    advance = prepare_linear_evolution(np.eye(2), np.ones(2))
    with pytest.raises(ValueError, match="initial"):
        list(advance(np.ones((2, 1)), [1]))


@pytest.mark.parametrize("method", LINEAR_EVOLUTION_METHODS)
def test_integer_initial_state_promotes_to_equation_dtype(method):
    """An integer initial condition must not truncate evolution."""
    xp = get_array_module()
    advance = prepare_linear_evolution(
        diagonal_linear_map(-xp.ones(1)),
        xp.ones(1),
        method=method,
        dt=0.1 if method == "euler" else None,
    )
    actual = next(advance(xp.array([2]), [0.1]))[..., 0]
    expected = 1.9 if method == "euler" else 1 + np.exp(-0.1)
    np.testing.assert_allclose(actual, [expected], rtol=1e-3)


@pytest.mark.parametrize("method", LINEAR_EVOLUTION_METHODS)
@pytest.mark.parametrize("batch_size", [1, 3, 32])
@pytest.mark.parametrize("kind", ["diagonal", "materialized", "matrix-free"])
def test_bounded_batches_preserve_samples_and_final_state(method, batch_size, kind):
    """Output chunks do not change an Euler grid or adaptive trajectory."""
    xp = get_array_module()
    rates = xp.array([-0.5, -2.0])
    if kind == "diagonal":
        A = diagonal_linear_map(rates)
    elif kind == "materialized":
        A = as_linear_map(xp.diag(rates))
    else:
        A = LinearMap(
            shape=(2, 2), dtype=float, matvec=lambda x: rates * x, rmatvec=lambda x: rates * x
        )
    b = xp.array([1.0, -1.0])
    initial = xp.array([3.0, 1.0])
    advance = prepare_linear_evolution(
        A, b, method=method, dt=0.1 if method == "euler" else None, rtol=1e-11, atol=1e-13
    )
    times = np.array([0.0, 0.03, 0.12, 0.29, 0.5, 0.71, 0.735])
    blocks = list(advance(initial, times, batch_size=batch_size))
    assert all(block.shape[0] == 2 and 1 <= block.shape[-1] <= batch_size for block in blocks)
    actual = np.concatenate(blocks, axis=-1)
    equilibrium = -np.asarray(b / rates)[:, None]
    if method == "euler":
        steps = np.floor(times / 0.1).astype(int)
        decay = (1 + 0.1 * np.asarray(rates)[:, None]) ** steps * (
            1 + np.asarray(rates)[:, None] * (times - 0.1 * steps)
        )
    else:
        decay = np.exp(np.asarray(rates)[:, None] * times)
    expected = equilibrium + (np.asarray(initial)[:, None] - equilibrium) * decay
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    final = next(advance(initial, [times[-1]], batch_size=1))[..., 0]
    np.testing.assert_allclose(actual[..., -1], final, rtol=1e-12, atol=1e-12)
    if kind == "matrix-free" and method == "euler":
        assert not A._dense_cache
    if kind == "diagonal" and method in ("euler", "exponential"):
        assert not A._dense_cache


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_batch_size(size):
    advance = prepare_linear_evolution(np.eye(1), np.ones(1))
    with pytest.raises(ValueError, match="batch_size"):
        list(advance(np.ones(1), [1], batch_size=size))


def test_scipy_failure_exposes_completed_partial_batch(monkeypatch):
    """A later failed step must not discard already sampled states."""
    original = linear_evolution.scipy.integrate.RK45

    class FailingRK45(original):
        def step(self):
            if self.t > 0.15:
                raise RuntimeError("injected failure")
            return super().step()

    monkeypatch.setattr(linear_evolution.scipy.integrate, "RK45", FailingRK45)
    advance = prepare_linear_evolution(-np.eye(1), np.zeros(1), method="RK45", rtol=1e-8)
    blocks = advance(np.ones(1), [0.0, 0.001, 0.01, 2.0], batch_size=4)
    partial = next(blocks)
    assert partial.shape == (1, 3)
    np.testing.assert_allclose(partial[0], np.exp(-np.array([0.0, 0.001, 0.01])), rtol=1e-6)
    with pytest.raises(RuntimeError, match="injected failure"):
        next(blocks)
