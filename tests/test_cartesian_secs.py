"""Tests for Cartesian elementary-current-system kernels."""

import numpy as np
import pytest

from kompe.secs import (
    cartesian_current_matrices,
    cartesian_magnetic_field_matrices,
)


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
def test_cartesian_current_matrices_have_finite_point_by_pole_shape(current_type):
    components = cartesian_current_matrices(
        x=[1.0, 2.0],
        y=[0.0, 1.0],
        z=[0.0, 0.0],
        x_poles=[0.0, -1.0, 1.0],
        y_poles=[2.0, -2.0, 3.0],
        z_poles=[0.0, 0.0, 0.0],
        current_type=current_type,
    )

    assert len(components) == 2
    assert all(component.shape == (2, 3) for component in components)
    assert all(np.isfinite(component).all() for component in components)


def test_cartesian_current_orientation_and_sheet_tolerance():
    curl_free_x, curl_free_y = cartesian_current_matrices(
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 11.0],
        [0.0],
        [0.0],
        [0.0],
        current_type="curl_free",
    )
    divergence_free_x, divergence_free_y = cartesian_current_matrices(
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        current_type="divergence_free",
    )

    np.testing.assert_allclose(curl_free_x[:, 0], [1.0 / (2 * np.pi), 0.0])
    np.testing.assert_allclose(curl_free_y, 0.0, atol=1e-16)
    np.testing.assert_allclose(divergence_free_x, 0.0, atol=1e-16)
    np.testing.assert_allclose(divergence_free_y[0, 0], -1.0 / (2 * np.pi))


def test_curl_free_magnetic_field_is_one_sided_across_current_sheet():
    bx, by, bz = cartesian_magnetic_field_matrices(
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, -1.0],
        [0.0],
        [0.0],
        [0.0],
        current_type="curl_free",
        constant=1.0,
    )

    np.testing.assert_allclose(bx, 0.0, atol=1e-16)
    np.testing.assert_allclose(by[:, 0], [-2.0, 0.0])
    np.testing.assert_allclose(bz, 0.0)


@pytest.mark.parametrize("kernel", [cartesian_current_matrices, cartesian_magnetic_field_matrices])
def test_cartesian_kernels_reject_unsupported_current_types(kernel):
    with pytest.raises(ValueError, match="current_type"):
        kernel([1.0], [0.0], [0.0], [0.0], [0.0], [0.0], current_type="potential")
