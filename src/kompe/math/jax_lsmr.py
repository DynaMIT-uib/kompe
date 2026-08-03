"""Small JAX LSMR implementation for internal least-squares solves.

Adapted from SciPy's ``lsmr`` implementation (BSD-3-Clause,
SciPy Developers) and the Fong/Saunders LSMR algorithm.
"""

from __future__ import annotations

from typing import Any


def lsmr(
    A: Any,
    b: Any,
    damp: float = 0.0,
    atol: float | None = 1e-6,
    btol: float | None = 1e-6,
    conlim: float | None = 1e8,
    maxiter: int | None = None,
    show: bool = False,
    x0: Any = None,
) -> tuple[Any, int, int, float, float, float, float, float]:
    """Solve ``min ||b - A x||`` using JAX array operations.

    This mirrors the core SciPy LSMR iteration for internal use with
    ``LinearMap`` objects. The iteration loop is intentionally
    controlled from Python so arbitrary map implementations can be used.

    Greek-derived recurrence names follow the Fong--Saunders algorithm;
    diagnostic names describe the quantities exposed in SciPy's return
    value. ``show`` is accepted for call compatibility, but this compact
    implementation does not print iteration logs.
    """
    del show

    import jax.numpy as jnp

    m, n = A.shape
    max_iterations = min(m, n) if maxiter is None else maxiter
    atol = 1e-6 if atol is None else atol
    btol = 1e-6 if btol is None else btol
    conlim = 1e8 if conlim is None else conlim

    b = jnp.atleast_1d(b).reshape(m)
    dtype = jnp.result_type(A.dtype, b, 1.0 if x0 is None else x0)
    u = b.astype(dtype)
    norm_b = _norm(u)

    # Initial Golub--Kahan bidiagonalization step.
    if x0 is None:
        x = jnp.zeros(n, dtype=dtype)
        beta = norm_b
    else:
        x = jnp.atleast_1d(jnp.asarray(x0, dtype=dtype)).reshape(n)
        u = u - A.matvec(x)
        beta = _norm(u)

    if beta > 0:
        u = u / beta
        v = A.rmatvec(u)
        alpha = _norm(v)
    else:
        v = jnp.zeros(n, dtype=dtype)
        alpha = 0.0

    if alpha > 0:
        v = v / alpha

    # Core solution recurrence.
    iteration = 0
    zeta_bar = alpha * beta
    alpha_bar = alpha
    rho = 1.0
    rho_bar = 1.0
    c_bar = 1.0
    s_bar = 0.0

    h = v
    h_bar = jnp.zeros(n, dtype=dtype)

    # Recurrence used to estimate ||r|| without forming the residual.
    beta_dd = beta
    beta_d = 0.0
    rho_d_old = 1.0
    tau_tilde_old = 0.0
    theta_tilde = 0.0
    zeta = 0.0
    residual_sum_squares = 0.0

    # Running operator-norm and condition-number estimates.
    norm_A_squared = alpha * alpha
    max_rho_bar = 0.0
    min_rho_bar = 1e100
    norm_A = norm_A_squared**0.5
    condition_A = 1.0
    norm_x = 0.0

    stop_code = 0
    condition_tolerance = 1.0 / conlim if conlim > 0 else 0.0
    norm_r = beta
    norm_Ar = alpha * beta
    if norm_Ar == 0:
        return x, stop_code, iteration, norm_r, norm_Ar, norm_A, condition_A, norm_x
    if norm_b == 0:
        x = jnp.zeros_like(x)
        return x, stop_code, iteration, norm_r, norm_Ar, norm_A, condition_A, norm_x

    while iteration < max_iterations:
        iteration += 1

        # Advance the Golub--Kahan bidiagonalization.
        u = -alpha * u + A.matvec(v)
        beta = _norm(u)
        if beta > 0:
            u = u / beta
            v = -beta * v + A.rmatvec(u)
            alpha = _norm(v)
            if alpha > 0:
                v = v / alpha

        # Eliminate damping and bidiagonal terms with stable rotations.
        c_hat, s_hat, alpha_hat = _sym_ortho(alpha_bar, damp)

        rho_old = rho
        c, s, rho = _sym_ortho(alpha_hat, beta)
        theta_new = s * alpha
        alpha_bar = c * alpha

        rho_bar_old = rho_bar
        zeta_old = zeta
        theta_bar = s_bar * rho
        rho_temp = c_bar * rho
        c_bar, s_bar, rho_bar = _sym_ortho(c_bar * rho, theta_new)
        zeta = c_bar * zeta_bar
        zeta_bar = -s_bar * zeta_bar

        # Update the solution through short vector recurrences.
        h_bar = -(theta_bar * rho / (rho_old * rho_bar_old)) * h_bar + h
        x = x + (zeta / (rho * rho_bar)) * h_bar
        h = -(theta_new / rho) * h + v

        # Update the residual-norm estimate.
        beta_acute = c_hat * beta_dd
        beta_check = -s_hat * beta_dd

        beta_hat = c * beta_acute
        beta_dd = -s * beta_acute

        theta_tilde_old = theta_tilde
        c_tilde_old, s_tilde_old, rho_tilde_old = _sym_ortho(rho_d_old, theta_bar)
        theta_tilde = s_tilde_old * rho_bar
        rho_d_old = c_tilde_old * rho_bar
        beta_d = -s_tilde_old * beta_d + c_tilde_old * beta_hat

        tau_tilde_old = (zeta_old - theta_tilde_old * tau_tilde_old) / rho_tilde_old
        tau_d = (zeta - theta_tilde * tau_tilde_old) / rho_d_old
        residual_sum_squares += beta_check * beta_check
        norm_r = (residual_sum_squares + (beta_d - tau_d) ** 2 + beta_dd * beta_dd) ** 0.5

        # Update ||A|| and cond(A) estimates.
        norm_A_squared += beta * beta
        norm_A = norm_A_squared**0.5
        norm_A_squared += alpha * alpha

        max_rho_bar = max(max_rho_bar, rho_bar_old)
        condition_A = max(max_rho_bar, rho_temp) / min(min_rho_bar, rho_temp)
        min_rho_bar = min(min_rho_bar, rho_bar)

        # Evaluate convergence using the estimates maintained above.
        norm_Ar = abs(zeta_bar)
        norm_x = _norm(x)

        relative_residual = norm_r / norm_b
        norm_A_times_r = norm_A * norm_r
        relative_normal_residual = (
            norm_Ar / norm_A_times_r if norm_A_times_r != 0 else float("inf")
        )
        inverse_condition = 1.0 / condition_A
        norm_ratio = norm_A * norm_x / norm_b
        backward_error = relative_residual / (1.0 + norm_ratio)
        residual_tolerance = btol + atol * norm_ratio

        stop_code = _stopping_code(
            iteration=iteration,
            max_iterations=max_iterations,
            relative_residual=relative_residual,
            relative_normal_residual=relative_normal_residual,
            inverse_condition=inverse_condition,
            backward_error=backward_error,
            residual_tolerance=residual_tolerance,
            least_squares_tolerance=atol,
            condition_tolerance=condition_tolerance,
        )
        if stop_code:
            break

    return x, stop_code, iteration, norm_r, norm_Ar, norm_A, condition_A, norm_x


def _stopping_code(
    *,
    iteration: int,
    max_iterations: int,
    relative_residual: float,
    relative_normal_residual: float,
    inverse_condition: float,
    backward_error: float,
    residual_tolerance: float,
    least_squares_tolerance: float,
    condition_tolerance: float,
) -> int:
    """Return the highest-priority satisfied SciPy LSMR stop code."""
    if relative_residual <= residual_tolerance:
        return 1
    if relative_normal_residual <= least_squares_tolerance:
        return 2
    if inverse_condition <= condition_tolerance:
        return 3
    if 1.0 + backward_error <= 1.0:
        return 4
    if 1.0 + relative_normal_residual <= 1.0:
        return 5
    if 1.0 + inverse_condition <= 1.0:
        return 6
    if iteration >= max_iterations:
        return 7
    return 0


def _norm(x: Any) -> float:
    import jax.numpy as jnp

    return float(jnp.linalg.norm(x))


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """Stable symmetric Givens rotation."""
    if b == 0:
        return _sign(a), 0.0, abs(a)
    if a == 0:
        return 0.0, _sign(b), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        s = _sign(b) / (1.0 + tau * tau) ** 0.5
        c = s * tau
        r = b / s
        return c, s, r
    tau = b / a
    c = _sign(a) / (1.0 + tau * tau) ** 0.5
    s = c * tau
    r = a / c
    return c, s, r


def _sign(value: float) -> float:
    return 1.0 if value >= 0 else -1.0
