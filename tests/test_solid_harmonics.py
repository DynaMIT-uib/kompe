"""Radial continuation and its backend-portable coefficient factors."""

import numpy as np
import pytest

from kompe import SHBasis, SolidHarmonicOperators
from kompe.math import backend_context


@pytest.mark.requires_jax
@pytest.mark.parametrize("branch", ["regular", "irregular"])
def test_radial_shifts_preserve_jax_radius_inputs_and_tracing(branch):
    """Radius-dependent calculations stay on the operand's backend."""
    import jax
    import jax.numpy as jnp

    radial = SolidHarmonicOperators(SHBasis(3, 2))
    factors = getattr(radial, f"{branch}_reference_shift_factors")
    operator = getattr(radial, f"{branch}_reference_shift_operator")
    powers = 1 - radial.basis.n if branch == "regular" else radial.basis.n + 2
    expected = (2.0 / 3.0) ** powers
    with backend_context("numpy"):
        start, end = jnp.asarray(2.0), jnp.asarray(3.0)
        result = factors(start, end)
        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(result, expected, rtol=1e-13)
        np.testing.assert_allclose(jax.jit(factors)(start, end), expected, rtol=1e-13)
        np.testing.assert_allclose(
            jax.jacfwd(factors)(start, end), expected * powers / 2.0, rtol=1e-13
        )
        shift = operator(start, end)
        assert shift.is_diagonal
        shifted = shift.matvec(jnp.ones(radial.basis.coefficient_count))
        assert isinstance(shifted, jax.Array)
        np.testing.assert_allclose(shifted, expected, rtol=1e-13)
