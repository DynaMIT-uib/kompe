"""Tests for SECS as a first-class spherical representation."""

import numpy as np
import pytest

from kompe import (
    Grid,
    RegionalCSGrid,
    RegionalCSProjection,
    ScalarSynthesis,
    SECSBasis,
    SurfaceOperators,
    is_secs_basis,
)
from kompe.secs import magnetic_field_matrices, surface_current_matrices


@pytest.fixture
def secs_basis():
    return SECSBasis(
        lat=[65.0, 70.0, 73.0],
        lon=[5.0, 18.0, 32.0],
        current_type="divergence_free",
    )


@pytest.fixture
def evaluation_grid():
    return Grid(lat=[61.0, 67.0, 72.0, 76.0], lon=[-2.0, 12.0, 24.0, 40.0])


def test_secs_is_scalar_synthesis_without_closed_surface_claims(secs_basis):
    assert isinstance(secs_basis, ScalarSynthesis)
    assert not isinstance(secs_basis, SurfaceOperators)
    assert is_secs_basis(secs_basis)
    assert secs_basis.kind == "SECS"
    assert secs_basis.index_length == 3
    assert secs_basis.index_names == ("latitude", "longitude")
    with pytest.raises(NotImplementedError, match="surface-current synthesis"):
        secs_basis.evaluate_on_grid(secs_basis.poles, derivative="theta")


def test_secs_accepts_regional_grid_for_poles_and_evaluation():
    regional = RegionalCSGrid(
        RegionalCSProjection((20.0, 70.0), 25.0),
        600.0,
        500.0,
        4,
        3,
        radius=6371.2,
    )
    basis = SECSBasis(poles=regional, current_type="curl_free")
    matrix = basis.evaluate_on_grid(regional)

    assert basis.index_length == regional.size
    assert matrix.shape == (regional.size, regional.size)


def test_secs_scalar_synthesis_has_explicit_physical_mode(evaluation_grid):
    poles = Grid(lat=[65.0, 70.0], lon=[5.0, 18.0])
    curl_free = SECSBasis(poles=poles, current_type="curl_free")
    divergence_free = SECSBasis(poles=poles, current_type="divergence_free")

    expected_potential = surface_current_matrices(
        evaluation_grid.lat,
        evaluation_grid.lon,
        poles.lat,
        poles.lon,
        current_type="potential",
        constant=curl_free.constant,
        RI=curl_free.radius,
    )
    expected_stream_function = surface_current_matrices(
        evaluation_grid.lat,
        evaluation_grid.lon,
        poles.lat,
        poles.lon,
        current_type="scalar",
        constant=divergence_free.constant,
        RI=divergence_free.radius,
    )

    np.testing.assert_allclose(curl_free.evaluate_on_grid(evaluation_grid), expected_potential)
    np.testing.assert_allclose(
        divergence_free.evaluate_on_grid(evaluation_grid), expected_stream_function
    )
    assert curl_free.signature != divergence_free.signature


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
def test_surface_current_kernel_matches_canonical_components(
    secs_basis, evaluation_grid, current_type
):
    east, north = surface_current_matrices(
        evaluation_grid.lat,
        evaluation_grid.lon,
        secs_basis.poles.lat,
        secs_basis.poles.lon,
        current_type=current_type,
        constant=secs_basis.constant,
        RI=secs_basis.radius,
    )
    canonical = secs_basis.get_surface_current_matrix(evaluation_grid, current_type=current_type)

    np.testing.assert_allclose(canonical[0], -north)
    np.testing.assert_allclose(canonical[1], east)
    operator = secs_basis.get_surface_current_operator(evaluation_grid, current_type=current_type)
    coefficients = np.array([0.4, -1.2, 0.7])
    np.testing.assert_allclose(
        operator @ coefficients,
        np.tensordot(canonical, coefficients, axes=1).reshape(-1),
    )


def test_two_component_secs_helmholtz_operator_matches_tensor(secs_basis, evaluation_grid):
    matrix = secs_basis.get_helmholtz_current_synthesis_matrix(evaluation_grid)
    operator = secs_basis.get_helmholtz_current_synthesis_operator(evaluation_grid)
    coefficients = np.array([[0.2, -0.5, 0.8], [1.0, 0.3, -0.4]])

    assert matrix.shape == (2, evaluation_grid.size, 2, secs_basis.index_length)
    np.testing.assert_allclose(
        operator @ coefficients.reshape(-1),
        np.tensordot(matrix, coefficients, axes=2).reshape(-1),
    )


def test_chunked_secs_current_operator_matches_dense_forward_and_adjoint(
    secs_basis, evaluation_grid
):
    dense = secs_basis.get_surface_current_operator(evaluation_grid)
    chunked = secs_basis.get_surface_current_operator(evaluation_grid, chunk_size=2)
    coefficients = np.array([0.4, -1.2, 0.7])
    values = np.linspace(-1.0, 1.0, 2 * evaluation_grid.size)
    coefficient_block = np.column_stack([coefficients, -2.0 * coefficients])
    value_block = np.column_stack([values, 0.5 * values])

    np.testing.assert_allclose(chunked @ coefficients, dense @ coefficients)
    np.testing.assert_allclose(chunked.rmatvec(values), dense.rmatvec(values))
    np.testing.assert_allclose(chunked.matmat(coefficient_block), dense.matmat(coefficient_block))
    np.testing.assert_allclose(chunked.rmatmat(value_block), dense.rmatmat(value_block))


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
def test_magnetic_field_uses_canonical_radial_theta_phi_order(
    secs_basis, evaluation_grid, current_type
):
    evaluation_radius = 6371.2e3
    east, north, radial = magnetic_field_matrices(
        evaluation_grid.lat,
        evaluation_grid.lon,
        evaluation_radius,
        secs_basis.poles.lat,
        secs_basis.poles.lon,
        current_type=current_type,
        constant=secs_basis.constant,
        RI=secs_basis.radius,
    )
    canonical = secs_basis.get_magnetic_field_matrix(
        evaluation_grid, evaluation_radius, current_type=current_type
    )
    np.testing.assert_allclose(canonical, np.stack([radial, -north, east]))


def test_secs_kernel_does_not_mutate_numpy_error_policy(secs_basis, evaluation_grid):
    before = np.geterr().copy()
    secs_basis.get_magnetic_field_matrix(evaluation_grid, 6371.2e3)
    assert np.geterr() == before


def test_secs_rejects_invalid_current_type(secs_basis, evaluation_grid):
    with pytest.raises(ValueError, match="current_type"):
        secs_basis.get_surface_current_matrix(evaluation_grid, current_type="unknown")
    with pytest.raises(ValueError, match="current_type"):
        secs_basis.get_magnetic_field_matrix(evaluation_grid, 6371.2e3, current_type="unknown")
