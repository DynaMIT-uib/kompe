"""Physical smoothness energies and their structured least-squares penalties."""

import numpy as np
import pytest
from scipy.linalg import null_space

from kompe import GlobalCSBasis, SHBasis, SphericalTransform
from kompe.basis import BasisSubset
from kompe.math import LeastSquaresSolver, LinearMap, get_array_module


@pytest.mark.parametrize("quadrupole", [False, True])
def test_cs_smoothness_converges_to_independent_spherical_energies(quadrupole):
    """Check cos(theta) and x²-y² against their exact full-sphere energies."""
    # mean(z²)=1/3, mean((x²-y²)²)=4/15; Laplacian eigenvalues are -2 and -6.
    eigenvalue, mean_square = (6, 4 / 15) if quadrupole else (2, 1 / 3)
    expected = np.array([eigenvalue * mean_square, 5 * eigenvalue**2 * mean_square])
    errors = []
    xp = get_array_module()
    for resolution in (4, 8, 16):
        basis = GlobalCSBasis(resolution)
        theta, phi = xp.deg2rad(basis.native_grid.theta), xp.deg2rad(basis.native_grid.phi)
        values = xp.sin(theta) ** 2 * xp.cos(2 * phi) if quadrupole else xp.cos(theta)
        scalar = basis.scalar_smoothness_operator()
        helmholtz = basis.helmholtz_smoothness_operator()
        actual = np.array(
            [
                float(xp.sum(xp.abs(scalar(values)) ** 2)),
                float(xp.sum(xp.abs(helmholtz(xp.stack([values, 2 * values]))) ** 2)),
            ]
        )
        errors.append(np.abs(actual - expected) / expected)
        assert scalar.materialized_matrix is None
        assert helmholtz.materialized_matrix is None
    assert np.all(errors[1] < 0.6 * errors[0])
    assert np.all(errors[2] < 0.6 * errors[1])
    assert np.all(errors[2] < 0.06)


@pytest.mark.parametrize("helmholtz", [False, True])
def test_sh_and_subset_smoothness_remain_vector_backed(helmholtz):
    """Analytical penalties do not need sampled derivatives or dense matrices."""
    parent = SHBasis(4, 3, mean_free=False)
    basis = BasisSubset(parent, [0, 2, 4, 7])
    name = "helmholtz_smoothness_operator" if helmholtz else "scalar_smoothness_operator"
    full, restricted = getattr(parent, name)(), getattr(basis, name)()
    assert full.is_diagonal and restricted.is_diagonal
    full_values = full.diagonal().reshape(full.input_shape)
    np.testing.assert_array_equal(
        restricted.diagonal().reshape(restricted.input_shape), full_values[..., [0, 2, 4, 7]]
    )
    assert full.materialized_matrix is None
    assert restricted.materialized_matrix is None


@pytest.mark.parametrize("helmholtz", [False, True])
def test_cs_subset_penalty_keeps_all_derivative_residuals(monkeypatch, helmholtz):
    """A coefficient subset does not imply dropping residuals elsewhere on the sphere."""
    import kompe.math.linear_map as module

    parent = GlobalCSBasis(4)
    indices = np.arange(0, parent.coefficient_count, 2)
    basis = BasisSubset(parent, indices)
    name = "helmholtz_smoothness_operator" if helmholtz else "scalar_smoothness_operator"
    full, restricted = getattr(parent, name)(), getattr(basis, name)()
    xp = get_array_module()
    values = np.random.default_rng(497).normal(size=restricted.input_shape + (2,))
    full_values = np.zeros(full.input_shape + (2,))
    full_values[..., indices, :] = values
    np.testing.assert_allclose(
        restricted(xp.asarray(values)), full(xp.asarray(full_values)), atol=1e-13
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Sparse subset normal diagonals must not use coefficient probes.")

    monkeypatch.setattr(module, "_normal_matrix_diag_from_matmat", unexpected)
    monkeypatch.setattr(LinearMap, "_dense_array", unexpected)
    expected = full.normal_matrix_diag().reshape(full.input_shape)[..., indices].reshape(-1)
    np.testing.assert_allclose(restricted.normal_matrix_diag(), expected)


@pytest.mark.parametrize("representation", ["scalar", "helmholtz"])
@pytest.mark.parametrize("method", ["normal_solve", "normal_pinv", "svd", "cgls", "lsmr"])
def test_cs_regularized_fit_matches_an_independent_augmented_problem(representation, method):
    """Check the weighted objective, relative scale, and exact potential gauges."""
    basis = GlobalCSBasis(4)
    strength = 0.03
    transform = SphericalTransform(
        basis, basis.native_grid, reg_lambda=strength, area_weighted=True
    )
    shape = (
        (basis.coefficient_count,) if representation == "scalar" else (2, basis.coefficient_count)
    )
    samples = np.random.default_rng(724).normal(size=shape + (2,))
    xp = get_array_module()
    actual = getattr(transform, f"analyze_{representation}")(
        xp.asarray(samples), solver=LeastSquaresSolver(method=method, tolerance=1e-12)
    )

    # Materialize only the independent reference, after the fit under test.
    data = getattr(transform, f"{representation}_synthesis_operator").to_matrix(backend="numpy")
    penalty = getattr(transform, f"{representation}_regularization_operator").to_matrix(
        backend="numpy"
    )
    weights = np.sqrt(
        np.tile(basis.native_grid.area_weights, 1 if representation == "scalar" else 2)
    )
    weighted_data = weights[:, None] * data
    data_diag = np.sum(weighted_data**2, axis=0)
    penalty_diag = np.sum(penalty**2, axis=0)
    scale = np.sqrt(
        strength * np.median(data_diag[data_diag > 0]) / np.median(penalty_diag[penalty_diag > 0])
    )
    coordinates = np.eye(data.shape[1])
    if representation == "helmholtz":
        mean = basis.scalar_mean_weights
        constraints = np.kron(np.eye(2), mean[None, :])
        coordinates = null_space(constraints)
    augmented = np.vstack([weighted_data @ coordinates, scale * penalty @ coordinates])
    rhs = np.vstack(
        [weights[:, None] * samples.reshape(data.shape[0], 2), np.zeros((penalty.shape[0], 2))]
    )
    expected = coordinates @ np.linalg.lstsq(augmented, rhs, rcond=1e-13)[0]
    np.testing.assert_allclose(actual, expected.reshape(shape + (2,)), rtol=2e-8, atol=2e-8)
