"""Polynomial-exact finite-difference stencil construction."""

from math import factorial

import numpy as np
import pytest

from kompe.math import finite_difference_weights


@pytest.mark.parametrize("points", [[-1, 0, 1], [0, 1, 2, 3], [-3, -1, 0, 2, 4]])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("spacing", [0.25, 2.0])
def test_stencil_differentiates_polynomials_at_the_origin(points, order, spacing):
    """Centered, one-sided, and uneven stencils preserve derivative units."""
    weights = finite_difference_weights(points, order=order, h=spacing)
    positions = spacing * np.asarray(points)
    for power in range(len(points)):
        expected = factorial(order) if power == order else 0.0
        assert weights @ positions**power == pytest.approx(expected, abs=1e-11)
