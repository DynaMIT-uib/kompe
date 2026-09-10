"""Sample constant-coefficient linear evolution, x' = A x + b."""

from functools import cache
from itertools import pairwise

import numpy as np
import scipy.integrate

from kompe.math.backend import get_array_module, to_numpy
from kompe.math.exponential import affine_exponential
from kompe.math.linear_map import as_linear_map

LINEAR_EVOLUTION_METHODS = (
    "euler",
    "exponential",
    "RK23",
    "RK45",
    "DOP853",
    "Radau",
    "BDF",
    "LSODA",
)


@cache
def _compiled_step(step, *, static_argnames=()):
    """Compile a stable array kernel, not each equation's data."""
    from jax import jit

    return jit(step, static_argnames=static_argnames)


def _euler_block(A, x, b, dt, steps):
    def step(_index, values):
        Ax = A * values if A.ndim == 1 else A @ values
        return values + dt * (Ax + b)

    if get_array_module(A, x, b) is np:
        for index in range(steps):
            x = step(index, x)
        return x

    from jax import lax

    return lax.fori_loop(0, steps, step, x)


def _affine_step(P, x, q):
    return (P * x if P.ndim == 1 else P @ x) + q


def _affine_samples(P, x, q, count):
    """Apply one propagator repeatedly, retaining a bounded sample block."""
    if get_array_module(P, x, q) is np:
        samples = []
        for _ in range(count):
            x = _affine_step(P, x, q)
            samples.append(x)
        return np.stack(samples, axis=-1)
    from jax import lax

    def step(state, _):
        state = _affine_step(P, state, q)
        return state, state

    _, samples = lax.scan(step, x, None, length=count)
    return samples.T


def _euler_samples(A, x, b, dt, steps, remainders):
    """Sample Euler without changing its accepted dt-spaced trajectory."""
    xp = get_array_module(A, x, b)

    def step(state, sampling):
        count, remainder = sampling
        state = _euler_block(A, state, b, dt, count)
        if xp is np:
            sampled = state if remainder == 0 else _euler_block(A, state, b, remainder, 1)
        else:
            from jax import lax

            sampled = lax.cond(
                remainder == 0, lambda: state, lambda: _euler_block(A, state, b, remainder, 1)
            )
        return state, sampled

    if xp is np:
        samples = []
        for sampling in zip(steps, remainders, strict=True):
            x, sampled = step(x, sampling)
            samples.append(sampled)
        return x, np.stack(samples, axis=-1)
    from jax import lax

    x, samples = lax.scan(step, x, (steps, remainders))
    return x, samples.T


def prepare_linear_evolution(A, b, *, method="exponential", dt=None, rtol=1e-3, atol=1e-6):
    """Prepare reusable sampled evolution of `x' = A x + b`.

    Return `advance(initial, times, *, batch_size=32, output_interval=None)`,
    an iterator of bounded arrays with shape ``A.input_shape + (n_samples,)``.
    Offsets from the initial state must be strictly increasing and non-negative.
    Batches do not restart accepted steps or the adaptive solver. Completed
    batches remain available if later integration fails or is interrupted.

    `A` and `b` are fixed throughout each call. Euler requires a positive
    `dt`; requested samples use its linear within-step interpolant without
    changing accepted steps. Exponential propagation needs no equilibrium
    and retains four recent duration maps. An optional `output_interval`
    supplies the exact regular cadence as a cache key: only differences
    within floating-point roundoff are matched to it.

    SciPy methods adapt over the whole interval and sample their dense
    interpolants. They materialize and transfer the fixed equation once
    at this explicit CPU boundary. `rtol` is relative; `atol` is a positive
    scalar or an array with the state shape, in the state's units. Euler
    preserves diagonal, materialized, and matrix-free operator execution;
    exponential propagation preserves diagonals and materializes other maps.
    """
    A = as_linear_map(A)
    if A.input_shape != A.output_shape:
        raise ValueError("A must have matching input and output shapes.")
    xp = get_array_module(*A.backend_operands, b)
    b = xp.asarray(b)
    if b.shape != A.input_shape:
        raise ValueError("b must have A's input shape.")
    if method not in LINEAR_EVOLUTION_METHODS:
        raise ValueError(f"Unknown linear evolution method: {method!r}.")
    if method == "euler":
        if dt is None or isinstance(dt, (bool, np.bool_)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and greater than zero.")
        dt = float(dt)
    elif dt is not None:
        raise ValueError("dt controls Euler stepping; omit it for other integrators.")

    forcing = b.reshape(-1)
    if method == "euler":
        array = A.diagonal() if A.is_diagonal else A.materialized_matrix
        if array is not None:
            array = xp.asarray(array)
            euler_samples = _euler_samples if xp is np else _compiled_step(_euler_samples)

        else:

            def euler_block(values, step_size, steps):
                def step(_index, x):
                    return x + step_size * (A.matvec(x) + forcing)

                if xp is np:
                    for index in range(steps):
                        values = step(index, values)
                    return values
                from jax import lax

                return lax.fori_loop(0, steps, step, values)

            if xp is not np:
                from jax import jit

                euler_block = jit(euler_block)

    elif method == "exponential":
        # Import locally: kompe.cache itself uses math.fingerprints.
        from kompe.cache import BoundedCache

        propagators = BoundedCache(4)
        exponential_samples = (
            _affine_samples
            if xp is np
            else _compiled_step(_affine_samples, static_argnames=("count",))
        )
    else:
        if isinstance(rtol, (bool, np.bool_)) or not np.isfinite(rtol) or rtol <= 0:
            raise ValueError("rtol must be finite and greater than zero.")
        # Transfer the complete fixed equation once, not each substep.
        matrix = A.to_matrix(backend="numpy")
        forcing_numpy = to_numpy(forcing)
        absolute_tolerance = to_numpy(atol)
        if (
            absolute_tolerance.shape not in ((), A.input_shape)
            or absolute_tolerance.dtype.kind == "b"
            or not np.all(np.isfinite(absolute_tolerance))
            or np.any(absolute_tolerance <= 0)
        ):
            raise ValueError("atol must be positive and scalar or have the state shape.")
        if absolute_tolerance.ndim:
            absolute_tolerance = absolute_tolerance.reshape(-1)

    def advance(initial, times, *, batch_size=32, output_interval=None):
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or times.size == 0
            or not np.all(np.isfinite(times))
            or times[0] < 0
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("times must be a nonempty, increasing array of non-negative offsets.")
        if output_interval is not None and (
            isinstance(output_interval, (bool, np.bool_))
            or not np.isfinite(output_interval)
            or output_interval <= 0
        ):
            raise ValueError("output_interval must be finite and greater than zero.")
        if (
            isinstance(batch_size, (bool, np.bool_))
            or int(batch_size) != batch_size
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        state = xp.asarray(initial)
        if state.shape != A.input_shape:
            raise ValueError("initial must have A's input shape.")
        state = xp.asarray(
            state, dtype=xp.result_type(state.dtype, A.dtype, b.dtype, 0.0)
        ).reshape(-1)
        step_index = 0
        previous_time = 0.0
        if method not in ("euler", "exponential") and times[-1] > 0:

            def rhs(_time, values):
                return matrix @ values + forcing_numpy

            options = {"rtol": rtol, "atol": absolute_tolerance}
            if method in ("Radau", "BDF"):
                options["jac"] = matrix
            elif method == "LSODA":
                options["jac"] = lambda _time, _values: matrix
            solver = getattr(scipy.integrate, method)(
                rhs, 0.0, to_numpy(state), times[-1], **options
            )

        for start in range(0, times.size, batch_size):
            batch_times = times[start : start + batch_size]
            if method == "euler" and array is not None:
                indices = np.floor(batch_times / dt).astype(int)
                steps = np.diff(np.r_[step_index, indices])
                remainders = batch_times - indices * dt
                state, samples = euler_samples(
                    array, state, forcing, dt, xp.asarray(steps), xp.asarray(remainders)
                )
                step_index = int(indices[-1])
            elif method == "exponential":
                durations = np.diff(np.r_[previous_time, batch_times])
                if output_interval is not None:
                    durations[np.abs(durations - output_interval) <= 4 * np.spacing(times[-1])] = (
                        output_interval
                    )
                boundaries = np.r_[
                    0, np.flatnonzero(durations[1:] != durations[:-1]) + 1, durations.size
                ]
                blocks = []
                for first, last in pairwise(boundaries):
                    duration = float(durations[first])
                    if duration == 0:
                        block = state[:, None]
                    else:
                        P, q = propagators.get_or_create(
                            duration, lambda duration=duration: affine_exponential(A, b, duration)
                        )
                        propagator = P.diagonal() if P.is_diagonal else P.materialized_matrix
                        block = exponential_samples(
                            propagator, state, q.reshape(-1), count=int(last - first)
                        )
                        state = block[:, -1]
                    blocks.append(block)
                samples = blocks[0] if len(blocks) == 1 else xp.concatenate(blocks, axis=-1)
            else:
                # Matrix-free Euler and SciPy retain their sequential numerical
                # execution. SciPy transfers an entire sample block, not each row.
                rows = []
                try:
                    for time in batch_times:
                        if time == 0:
                            sampled = state if method == "euler" else to_numpy(state)
                        elif method == "euler":
                            index = int(np.floor(time / dt))
                            if index > step_index:
                                state = euler_block(state, dt, index - step_index)
                                step_index = index
                            remainder = time - index * dt
                            sampled = state if remainder == 0 else euler_block(state, remainder, 1)
                        else:
                            while solver.t < time:
                                message = solver.step()
                                if solver.status == "failed":
                                    raise RuntimeError(
                                        f"SciPy integrator {method!r} failed: {message}"
                                    )
                                interpolant = solver.dense_output()
                            sampled = solver.y.copy() if time == solver.t else interpolant(time)
                        rows.append(sampled)
                except (Exception, KeyboardInterrupt):
                    # Expose completed samples, then propagate the original failure.
                    if rows:
                        samples = (
                            xp.stack(rows, axis=-1)
                            if method == "euler"
                            else xp.asarray(np.stack(rows, axis=-1))
                        )
                        yield samples.reshape(A.input_shape + (len(rows),))
                    raise
                samples = (
                    xp.stack(rows, axis=-1)
                    if method == "euler"
                    else xp.asarray(np.stack(rows, axis=-1))
                )
            previous_time = float(batch_times[-1])
            yield samples.reshape(A.input_shape + (batch_times.size,))

    return advance


__all__ = ["LINEAR_EVOLUTION_METHODS", "prepare_linear_evolution"]
