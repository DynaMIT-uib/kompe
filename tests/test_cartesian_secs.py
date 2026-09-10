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


def test_cartesian_df_field_on_axis_matches_biot_savart_ring_integral():
    """Integrating circular sheet currents gives Bz=-mu0/(4 pi |z|)."""
    from kompe.constants import MU0

    z = np.array([-2.0, 3.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        bx, by, bz = cartesian_magnetic_field_matrices(0.0, 0.0, z, 0.0, 0.0, 0.0)
    np.testing.assert_array_equal(bx, 0.0)
    np.testing.assert_array_equal(by, 0.0)
    # A ring of width dr carries dI=-dr/(2 pi r). Its axis field is
    # mu0*dI*r**2/(2*(r**2+z**2)**1.5); integral_0^inf = -mu0/(4 pi |z|).
    np.testing.assert_allclose(bz[:, 0], -MU0 / (4 * np.pi * abs(z)), rtol=1e-14)


def test_cartesian_df_field_preserves_near_axis_slope():
    """The finite horizontal limit must not be lost by cancellation."""
    x = 1e-10
    bx, by, _ = cartesian_magnetic_field_matrices(x, 0.0, [2.0, -2.0], 0.0, 0.0, 0.0, constant=1.0)
    np.testing.assert_allclose(bx[:, 0] / x, [-1 / 8, 1 / 8], rtol=1e-14)
    np.testing.assert_array_equal(by, 0.0)


def test_cartesian_cf_field_does_not_move_the_current_sheet():
    """The exact jump location is z=0, not an isclose-sized layer."""
    with np.errstate(invalid="ignore", divide="ignore"):
        below = cartesian_magnetic_field_matrices(
            [1.0, 0.0], 0.0, -1e-10, 0.0, 0.0, 0.0, current_type="curl_free"
        )
    np.testing.assert_array_equal(below, np.zeros((3, 2, 1)))


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
def test_cartesian_fields_obey_ampere_sheet_jump(current_type):
    """z-hat cross (B+ - B-) equals mu0 times the surface current."""
    from kompe.constants import MU0

    x, y = np.array([1.0, -2.0]), np.array([2.0, -1.0])
    current = np.stack(cartesian_current_matrices(x, y, 0, 0, 0, 0, current_type=current_type))
    above = np.asarray(
        cartesian_magnetic_field_matrices(x, y, 1e-9, 0, 0, 0, current_type=current_type)
    )
    below = np.asarray(
        cartesian_magnetic_field_matrices(x, y, -1e-9, 0, 0, 0, current_type=current_type)
    )
    jump = above - below
    np.testing.assert_allclose(np.stack([-jump[1], jump[0]]), MU0 * current, rtol=1e-8, atol=1e-22)


@pytest.mark.parametrize("kernel", [cartesian_current_matrices, cartesian_magnetic_field_matrices])
def test_cartesian_kernels_reject_unsupported_current_types(kernel):
    with pytest.raises(ValueError, match="current_type"):
        kernel([1.0], [0.0], [0.0], [0.0], [0.0], [0.0], current_type="potential")


@pytest.mark.requires_jax
@pytest.mark.parametrize("kernel", [cartesian_current_matrices, cartesian_magnetic_field_matrices])
def test_cartesian_kernels_preserve_jax_arrays(kernel):
    """Continuous CECS calculations stay on the input array backend."""
    import jax.numpy as jnp

    components = kernel(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, -1.0]),
        jnp.asarray([2.0, -2.0]),
        jnp.asarray([0.0, 0.0]),
    )

    assert all("jax" in type(component).__module__ for component in components)
