"""Device-side least-squares iterations for JAX-compatible linear maps.

The LSMR recurrence follows SciPy (BSD-3-Clause, SciPy Developers) and
Fong/Saunders. Each RHS has its own convergence test, also in batched solves.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from kompe.math.linear_map import LinearMap


class _LSMRState(NamedTuple):
    """Golub--Kahan vectors, scalar recurrences, and convergence estimates."""

    x: Any
    u: Any
    v: Any
    h: Any
    h_bar: Any
    alpha: Any
    alpha_bar: Any
    rho: Any
    rho_bar: Any
    c_bar: Any
    s_bar: Any
    beta_dd: Any
    beta_d: Any
    rho_d: Any
    tau_tilde: Any
    theta_tilde: Any
    zeta: Any
    zeta_bar: Any
    residual_sum_squares: Any
    norm_A_squared: Any
    max_rho_bar: Any
    min_rho_bar: Any
    iteration: Any
    stop_code: Any
    norm_r: Any
    norm_Ar: Any
    norm_A: Any
    condition_A: Any
    norm_x: Any


def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=None, x0=None):
    """Solve ``min ||b - A x||² + damp² ||x - x0||²``, as in SciPy LSMR.

    Return SciPy's eight solution/diagnostic values as JAX arrays. The whole
    recurrence runs in ``lax.while_loop``; A's actions must be JIT-compatible.
    This is an iterative solve, not a fixed linear inverse with an adjoint.
    """
    m, n = A.shape
    max_iterations = min(m, n) if maxiter is None else maxiter
    atol = 1e-6 if atol is None else atol
    btol = 1e-6 if btol is None else btol
    conlim = 1e8 if conlim is None else conlim
    condition_tolerance = jnp.where(conlim > 0, 1.0 / jnp.where(conlim > 0, conlim, 1), 0)

    b = jnp.asarray(b).reshape(m)
    dtype = jnp.result_type(A.dtype, b, 1.0 if x0 is None else x0)
    u = b.astype(dtype)
    norm_b = jnp.linalg.norm(u)
    x = jnp.zeros(n, dtype=dtype) if x0 is None else jnp.asarray(x0, dtype=dtype).reshape(n)
    if x0 is not None:
        u = u - A.matvec(x)

    # Initial Golub--Kahan bidiagonalization. Zero norms leave zero vectors.
    beta = jnp.linalg.norm(u)
    u = u / jnp.where(beta > 0, beta, 1)
    v = A.rmatvec(u)
    alpha = jnp.linalg.norm(v)
    v = v / jnp.where(alpha > 0, alpha, 1)
    zero = jnp.zeros_like(alpha)
    one = jnp.ones_like(alpha)
    x = jnp.where((norm_b == 0) & (alpha * beta != 0), jnp.zeros_like(x), x)
    initial = _LSMRState(
        x=x,
        u=u,
        v=v,
        h=v,
        h_bar=jnp.zeros_like(x),
        alpha=alpha,
        alpha_bar=alpha,
        rho=one,
        rho_bar=one,
        c_bar=one,
        s_bar=zero,
        beta_dd=beta,
        beta_d=zero,
        rho_d=one,
        tau_tilde=zero,
        theta_tilde=zero,
        zeta=zero,
        zeta_bar=alpha * beta,
        residual_sum_squares=zero,
        norm_A_squared=alpha**2,
        max_rho_bar=zero,
        min_rho_bar=jnp.full_like(alpha, jnp.inf),
        iteration=jnp.int32(0),
        stop_code=jnp.int32(0),
        norm_r=beta,
        norm_Ar=alpha * beta,
        norm_A=alpha,
        condition_A=one,
        norm_x=zero,
    )

    def unfinished(s):
        return (
            (s.iteration < max_iterations) & (s.stop_code == 0) & (s.norm_Ar != 0) & (norm_b != 0)
        )

    def step(s):
        # Advance the Golub--Kahan bidiagonalization.
        u = A.matvec(s.v) - s.alpha * s.u
        beta = jnp.linalg.norm(u)
        u = u / jnp.where(beta > 0, beta, 1)

        def advance_v():
            v = A.rmatvec(u) - beta * s.v
            alpha = jnp.linalg.norm(v)
            return alpha, v / jnp.where(alpha > 0, alpha, 1)

        alpha, v = jax.lax.cond(beta > 0, advance_v, lambda: (s.alpha, s.v))

        # Eliminate damping and bidiagonal terms with stable rotations.
        c_hat, s_hat, alpha_hat = _sym_ortho(s.alpha_bar, damp)
        c, sine, rho = _sym_ortho(alpha_hat, beta)
        theta_new = sine * alpha
        alpha_bar = c * alpha
        theta_bar = s.s_bar * rho
        rho_temp = s.c_bar * rho
        c_bar, s_bar, rho_bar = _sym_ortho(rho_temp, theta_new)
        zeta = c_bar * s.zeta_bar
        zeta_bar = -s_bar * s.zeta_bar

        # Update the solution through short vector recurrences.
        h_bar = s.h - (theta_bar * rho / (s.rho * s.rho_bar)) * s.h_bar
        x = s.x + (zeta / (rho * rho_bar)) * h_bar
        h = v - (theta_new / rho) * s.h

        # Estimate ||r|| without forming the residual.
        beta_acute = c_hat * s.beta_dd
        beta_check = -s_hat * s.beta_dd
        beta_hat = c * beta_acute
        beta_dd = -sine * beta_acute
        c_tilde, s_tilde, rho_tilde = _sym_ortho(s.rho_d, theta_bar)
        theta_tilde = s_tilde * rho_bar
        rho_d = c_tilde * rho_bar
        beta_d = -s_tilde * s.beta_d + c_tilde * beta_hat
        tau_tilde = (s.zeta - s.theta_tilde * s.tau_tilde) / rho_tilde
        tau_d = (zeta - theta_tilde * tau_tilde) / rho_d
        residual_sum_squares = s.residual_sum_squares + beta_check**2
        norm_r = jnp.sqrt(residual_sum_squares + (beta_d - tau_d) ** 2 + beta_dd**2)

        # Operator norm, condition number, and SciPy's stopping criteria.
        norm_A_squared = s.norm_A_squared + beta**2
        norm_A = jnp.sqrt(norm_A_squared)
        norm_A_squared = norm_A_squared + alpha**2
        max_rho_bar = jnp.maximum(s.max_rho_bar, s.rho_bar)
        condition_A = jnp.maximum(max_rho_bar, rho_temp) / jnp.minimum(s.min_rho_bar, rho_temp)
        min_rho_bar = jnp.minimum(s.min_rho_bar, rho_bar)
        norm_Ar = jnp.abs(zeta_bar)
        norm_x = jnp.linalg.norm(x)
        relative_residual = norm_r / norm_b
        norm_A_times_r = norm_A * norm_r
        relative_normal_residual = jnp.where(
            norm_A_times_r != 0, norm_Ar / norm_A_times_r, jnp.inf
        )
        inverse_condition = 1.0 / condition_A
        norm_ratio = norm_A * norm_x / norm_b
        backward_error = relative_residual / (1.0 + norm_ratio)
        residual_tolerance = btol + atol * norm_ratio
        iteration = s.iteration + 1
        # First satisfied condition has priority, exactly as in SciPy.
        stop_code = jnp.select(
            [
                relative_residual <= residual_tolerance,
                relative_normal_residual <= atol,
                inverse_condition <= condition_tolerance,
                1 + backward_error <= 1,
                1 + relative_normal_residual <= 1,
                1 + inverse_condition <= 1,
                iteration >= max_iterations,
            ],
            [1, 2, 3, 4, 5, 6, 7],
            default=0,
        ).astype(jnp.int32)
        return _LSMRState(
            x=x,
            u=u,
            v=v,
            h=h,
            h_bar=h_bar,
            alpha=alpha,
            alpha_bar=alpha_bar,
            rho=rho,
            rho_bar=rho_bar,
            c_bar=c_bar,
            s_bar=s_bar,
            beta_dd=beta_dd,
            beta_d=beta_d,
            rho_d=rho_d,
            tau_tilde=tau_tilde,
            theta_tilde=theta_tilde,
            zeta=zeta,
            zeta_bar=zeta_bar,
            residual_sum_squares=residual_sum_squares,
            norm_A_squared=norm_A_squared,
            max_rho_bar=max_rho_bar,
            min_rho_bar=min_rho_bar,
            iteration=iteration,
            stop_code=stop_code,
            norm_r=norm_r,
            norm_Ar=norm_Ar,
            norm_A=norm_A,
            condition_A=condition_A,
            norm_x=norm_x,
        )

    s = jax.lax.while_loop(unfinished, step, initial)
    return s.x, s.stop_code, s.iteration, s.norm_r, s.norm_Ar, s.norm_A, s.condition_A, s.norm_x


def _sym_ortho(a, b):
    """Stable symmetric Givens rotation, including the all-zero case."""
    radius = jnp.hypot(a, b)
    denominator = jnp.where(radius != 0, radius, 1)
    return jnp.where(radius == 0, 1, a / denominator), b / denominator, radius


def solve_lsmr_columns(A, rhs, *, preconditioner=None, maxiter=None, damp=0.0, x0=None, **options):
    """Compile independent LSMR solves and reuse them across RHS blocks."""
    operator = A if preconditioner is None else A @ preconditioner
    if x0 is not None:
        rhs = rhs - A.matmat(x0)

    def solve_column(b):
        # Same unit normalization as the NumPy solver. Only recurrence
        # scalars change per RHS; the original operator is never copied.
        norm_b = jnp.linalg.norm(b)
        scale = jnp.linalg.norm(operator.rmatvec(b / jnp.where(norm_b > 0, norm_b, 1)))
        scale = jnp.where(scale > 0, scale, 1)
        normalized = LinearMap(
            shape=operator.shape,
            dtype=operator.dtype,
            matvec=lambda x: operator.matvec(x) / scale,
            rmatvec=lambda x: operator.rmatvec(x) / scale,
        )
        x, stop_code, *_ = lsmr(
            normalized, b / scale, damp=damp / scale, maxiter=maxiter, **options
        )
        return x, stop_code

    solution, codes = jax.vmap(solve_column, in_axes=1, out_axes=(1, 0))(rhs)
    if preconditioner is not None:
        solution = preconditioner.matmat(solution)
    return (solution if x0 is None else x0 + solution), codes


def solve_cgls_columns(A, rhs, *, preconditioner=None, maxiter=None, x0=None, tol=1e-5, atol=0.0):
    """Apply JAX CG to A* A with independent convergence for every RHS."""
    normal_rhs = A.rmatmat(rhs)

    def solve_column(b, initial):
        x, _ = cg(
            lambda x: A.rmatvec(A.matvec(x)),
            b,
            M=None if preconditioner is None else preconditioner.matvec,
            maxiter=maxiter,
            x0=initial,
            tol=tol,
            atol=atol,
        )
        return x

    solution = jax.vmap(solve_column, in_axes=(1, None if x0 is None else 1), out_axes=1)(
        normal_rhs, x0
    )
    # JAX CG currently returns no convergence status. Check the actual
    # normal residual once per RHS block, without per-iteration host work.
    residual = normal_rhs - A.rmatmat(A.matmat(solution))
    target = jnp.maximum(atol, tol * jnp.linalg.norm(normal_rhs, axis=0))
    converged = jnp.linalg.norm(residual, axis=0) <= target
    return solution, converged
