"""Tests for the LinearMap abstraction."""

from itertools import product

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from kompe.math import (
    einsum_linear_map,
    einsum_linear_map_from_matvec,
    get_array_module,
    jax_enabled,
    set_backend,
)
from kompe.math.least_squares_problem import LeastSquaresProblem
from kompe.math.least_squares_solver import LeastSquaresSolver
from kompe.math.linear_map import (
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
    identity_linear_map,
    is_identity_linear_map,
    pointwise_matrix_linear_map,
    take_linear_map,
    vstack_linear_maps,
)

# Core maps and shaped operations


def test_dense_linear_map_matches_matrix_operations():
    """Dense maps match matrix operations."""
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    other = np.array([[2.0, -1.0], [0.5, 4.0]])
    x = np.array([0.25, -2.0])
    y = np.array([1.0, -3.0, 2.0])
    block = np.column_stack([x, x + 1.0])

    linear_map = as_linear_map(matrix)

    np.testing.assert_allclose(linear_map.matvec(x), matrix @ x)
    np.testing.assert_allclose(linear_map.rmatvec(y), matrix.T @ y)
    np.testing.assert_allclose(linear_map.matmat(block), matrix @ block)
    np.testing.assert_allclose(
        (linear_map @ as_linear_map(other)).to_matrix(backend="numpy"), matrix @ other
    )


def test_array_linear_map_preserves_scalar_input_shape():
    """An explicit zero-dimensional input shape is not replaced by inference."""
    matrix = np.array([[2.0], [3.0]])
    linear_map = as_linear_map(matrix, input_shape=(), output_shape=(2,))

    assert linear_map.input_shape == ()
    np.testing.assert_allclose(linear_map.matvec(np.array(4.0)), [8.0, 12.0])


def test_linear_map_conversion_does_not_hide_array_errors():
    """Errors raised while producing an array retain their original meaning."""

    class BrokenArray:
        def __array__(self, dtype=None, copy=None):
            raise RuntimeError("array construction failed")

    with pytest.raises(RuntimeError, match="array construction failed"):
        as_linear_map(BrokenArray())


def test_linear_map_addition_matches_matrix_operations():
    """Linear maps add through the same operator interface."""
    left = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    right = np.array([[0.5, -1.0], [2.0, 0.25], [-3.0, 4.0]])
    linear_map = as_linear_map(left) + as_linear_map(right)
    x = np.array([0.25, -2.0])
    y = np.array([1.0, -3.0, 2.0])
    block = np.column_stack([x, x + 1.0])
    expected = left + right

    np.testing.assert_allclose(linear_map.matvec(x), expected @ x)
    np.testing.assert_allclose(linear_map.rmatvec(y), expected.T @ y)
    np.testing.assert_allclose(linear_map.matmat(block), expected @ block)
    np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), expected)
    np.testing.assert_allclose(linear_map.normal_matrix_diag(), np.sum(expected**2, axis=0))


def test_linear_map_addition_requires_matching_shape_metadata():
    """Linear map addition rejects mismatched shaped domains."""
    matrix = np.eye(4)
    left = as_linear_map(matrix, input_shape=(2, 2), output_shape=(2, 2))
    right = as_linear_map(matrix, input_shape=(4,), output_shape=(4,))

    with pytest.raises(ValueError, match="Shape metadata mismatch"):
        _ = left + right


def test_vstack_linear_maps_matches_stacked_matrix_operations():
    """Vertically stacked maps apply and adjoint as one map."""
    top = np.array([[1.0, 2.0], [3.0, 5.0]])
    bottom = np.array([[7.0, 11.0]])
    stacked = vstack_linear_maps([as_linear_map(top), as_linear_map(bottom)])
    expected = np.vstack([top, bottom])
    x = np.array([0.25, -2.0])
    y = np.array([1.0, -3.0, 2.0])
    block = np.column_stack([x, x + 1.0])

    np.testing.assert_allclose(stacked.matvec(x), expected @ x)
    np.testing.assert_allclose(stacked.rmatvec(y), expected.T @ y)
    np.testing.assert_allclose(stacked.matmat(block), expected @ block)
    np.testing.assert_allclose(stacked.to_matrix(backend="numpy"), expected)
    np.testing.assert_allclose(stacked.normal_matrix_diag(), np.sum(expected**2, axis=0))
    assert stacked.input_shape == (2,)
    assert stacked.output_shape == (3,)


def test_vstack_linear_maps_elides_single_row():
    """Single-row stacks preserve the underlying map structure."""
    matrix = as_linear_map(np.array([[1.0, 2.0], [3.0, 5.0]]), output_shape=(2,), input_shape=(2,))

    assert vstack_linear_maps([matrix], input_shape=(2,)) is matrix


def test_vstack_linear_maps_empty_stack_has_stable_adjoint_shapes():
    """Empty stacks have well-defined batched adjoints."""
    stacked = vstack_linear_maps([], input_shape=(2,))

    assert stacked.shape == (0, 2)
    np.testing.assert_allclose(stacked.matvec(np.ones(2)), np.zeros(0))
    np.testing.assert_allclose(stacked.rmatvec(np.zeros(0)), np.zeros(2))
    np.testing.assert_allclose(stacked.matmat(np.ones((2, 3))), np.zeros((0, 3)))
    np.testing.assert_allclose(stacked.rmatmat(np.zeros((0, 3))), np.zeros((2, 3)))
    np.testing.assert_allclose(stacked.rmatmat(np.zeros(0)), np.zeros((2, 1)))


def test_linear_map_shape_metadata_is_validated_and_relabelable():
    """LinearMap shape metadata is strict and can be relabeled."""
    matrix = np.arange(12.0).reshape(3, 4)
    linear_map = as_linear_map(matrix, input_shape=(2, 2), output_shape=(3,))
    relabeled = as_linear_map(linear_map, input_shape=(4,), output_shape=(3,))

    assert linear_map.input_shape == (2, 2)
    assert linear_map.output_shape == (3,)
    assert relabeled.input_shape == (4,)
    assert relabeled.output_shape == (3,)
    np.testing.assert_allclose(relabeled.to_matrix(backend="numpy"), matrix)

    with pytest.raises(ValueError, match="Input shape"):
        as_linear_map(linear_map, input_shape=(5,))


def test_relabeling_linear_map_preserves_dense_materialization():
    """Shape labels do not discard an already materialized flat matrix."""
    matrix = np.arange(16.0).reshape(4, 4)
    linear_map = as_linear_map(matrix, input_shape=(4,), output_shape=(4,))
    materialized = linear_map.to_matrix(backend="numpy")

    relabeled = as_linear_map(linear_map, input_shape=(2, 2), output_shape=(2, 2))

    assert relabeled.to_matrix(backend="numpy") is materialized
    assert relabeled.to_array().shape == (2, 2, 2, 2)
    np.testing.assert_allclose(relabeled.to_array().reshape(4, 4), matrix)


def test_diagonal_linear_map_matches_dense_diagonal():
    """Diagonal helper matches dense diagonal application."""
    diag = diagonal_linear_map(np.array([2.0, 3.0]))
    expected = np.diag([2.0, 3.0])
    x = np.arange(2.0)

    np.testing.assert_allclose(diag.matvec(x), expected @ x)
    np.testing.assert_allclose(diag.to_matrix(backend="numpy"), expected)
    np.testing.assert_allclose(diag.diagonal(backend="numpy"), [2.0, 3.0])


def test_materialized_diagonal_map_keeps_vector_application():
    """A full matrix requested for inspection does not replace diagonal application."""
    diagonal = diagonal_linear_map(np.array([2.0, 3.0]))
    calls = []
    original_matvec = diagonal._matvec
    original_rmatvec = diagonal._rmatvec
    original_matmat = diagonal._matmat
    original_rmatmat = diagonal._rmatmat
    object.__setattr__(
        diagonal, "_matvec", lambda values: calls.append("matvec") or original_matvec(values)
    )
    object.__setattr__(
        diagonal, "_rmatvec", lambda values: calls.append("rmatvec") or original_rmatvec(values)
    )
    object.__setattr__(
        diagonal, "_matmat", lambda values: calls.append("matmat") or original_matmat(values)
    )
    object.__setattr__(
        diagonal, "_rmatmat", lambda values: calls.append("rmatmat") or original_rmatmat(values)
    )

    diagonal.to_matrix(backend="numpy")
    vector = np.array([5.0, 7.0])
    block = np.eye(2)
    diagonal.matvec(vector)
    diagonal.rmatvec(vector)
    diagonal.matmat(block)
    diagonal.rmatmat(block)

    assert calls == ["matvec", "rmatvec", "matmat", "rmatmat"]


def test_identity_linear_map_is_noop_without_dense_materialization():
    """Identity maps apply without storing an explicit dense matrix."""
    identity = identity_linear_map((2, 2))
    vector = np.arange(4.0)
    block = np.column_stack([vector, vector + 1.0])

    assert is_identity_linear_map(identity)
    np.testing.assert_allclose(identity.matvec(vector), vector)
    np.testing.assert_allclose(identity.rmatvec(vector), vector)
    np.testing.assert_allclose(identity.matmat(block), block)
    assert identity._dense_cache == {}
    np.testing.assert_allclose(identity.to_matrix(backend="numpy"), np.eye(4))


def test_identity_linear_map_composition_is_elided():
    """Composing with identity returns the other map directly."""
    matrix = as_linear_map(np.array([[1.0, 2.0], [3.0, 5.0]]))
    identity = identity_linear_map((2,))

    assert identity @ matrix is matrix
    assert matrix @ identity is matrix


def test_take_linear_map_selects_axis_and_scatter_adjoint():
    """Axis take maps gather forward and scatter through the adjoint."""
    values = np.arange(12.0).reshape(3, 4)
    indices = np.array([0, 2, 2])
    linear_map = take_linear_map(values.shape, indices, axis=1)
    block = np.column_stack([values.reshape(-1), values.reshape(-1) + 1.0])
    output = values[:, indices]
    adjoint_input = np.arange(output.size, dtype=float).reshape(output.shape)
    expected_adjoint = np.zeros_like(values)
    np.add.at(expected_adjoint, (slice(None), indices), adjoint_input)
    adjoint_block = np.column_stack([adjoint_input.reshape(-1), adjoint_input.reshape(-1) + 1.0])
    duplicate_counts = np.broadcast_to([1.0, 0.0, 2.0, 0.0], values.shape)

    assert linear_map.input_shape == values.shape
    assert linear_map.output_shape == output.shape
    np.testing.assert_allclose(linear_map.matvec(values), output.reshape(-1))
    np.testing.assert_allclose(
        linear_map.matmat(block),
        np.column_stack([output.reshape(-1), (values[:, indices] + 1.0).reshape(-1)]),
    )
    np.testing.assert_allclose(linear_map.rmatvec(adjoint_input), expected_adjoint.reshape(-1))
    np.testing.assert_allclose(
        linear_map.rmatmat(adjoint_block),
        np.column_stack(
            [expected_adjoint.reshape(-1), (expected_adjoint + duplicate_counts).reshape(-1)]
        ),
    )
    np.testing.assert_allclose(linear_map.normal_matrix_diag(), duplicate_counts.reshape(-1))
    expected_dense = np.zeros((output.size, values.size))
    input_indices = np.arange(values.size).reshape(values.shape)[:, indices].reshape(-1)
    expected_dense[np.arange(output.size), input_indices] = 1.0
    np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), expected_dense)


def test_take_linear_map_accepts_boolean_mask():
    """Boolean masks are accepted for one selected axis."""
    values = np.arange(6.0).reshape(2, 3)
    linear_map = take_linear_map(values.shape, [True, False, True], axis=-1)

    np.testing.assert_allclose(linear_map.matvec(values), values[:, [0, 2]].reshape(-1))


def test_pointwise_matrix_linear_map_matches_local_component_transform():
    """Pointwise component maps apply local matrices and adjoints."""
    matrix = (
        np.arange(24.0).reshape(2, 3, 4) / 10.0
        + 1j * np.arange(24.0, 48.0).reshape(2, 3, 4) / 20.0
    )
    values = np.arange(12.0).reshape(3, 4) / 5.0
    linear_map = pointwise_matrix_linear_map(matrix)
    expected = np.einsum("abg,bg->ag", matrix, values)

    adjoint_input = np.arange(8.0).reshape(2, 4) / 7.0
    expected_adjoint = np.einsum("abg,ag->bg", matrix.conj(), adjoint_input)
    block = np.column_stack([values.reshape(-1), values.reshape(-1) + 1.0])
    expected_block = np.stack([expected, np.einsum("abg,bg->ag", matrix, values + 1.0)], axis=-1)

    assert linear_map.input_shape == values.shape
    assert linear_map.output_shape == expected.shape
    np.testing.assert_allclose(linear_map.matvec(values), expected.reshape(-1))
    np.testing.assert_allclose(linear_map.rmatvec(adjoint_input), expected_adjoint.reshape(-1))
    np.testing.assert_allclose(linear_map.matmat(block), expected_block.reshape(expected.size, 2))
    np.testing.assert_allclose(
        linear_map.normal_matrix_diag(), np.sum(np.abs(matrix) ** 2, axis=0).reshape(-1)
    )
    expected_dense = np.zeros((expected.size, values.size), dtype=matrix.dtype)
    for output_component in range(matrix.shape[0]):
        output_rows = output_component * matrix.shape[2] + np.arange(matrix.shape[2])
        for input_component in range(matrix.shape[1]):
            input_cols = input_component * matrix.shape[2] + np.arange(matrix.shape[2])
            expected_dense[output_rows, input_cols] = matrix[output_component, input_component]
    np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), expected_dense)


@pytest.mark.requires_jax
def test_structured_dense_builders_preserve_jax_backend(monkeypatch):
    """New structured maps should materialize matrices on JAX."""
    import jax.numpy as jnp

    import kompe.math.linear_map as linear_map_module

    previous_backend = jax_enabled()
    matrix = jnp.arange(24.0).reshape(2, 3, 4)
    pointwise = pointwise_matrix_linear_map(matrix)
    selector = take_linear_map((2, 4), [0, 2], axis=1)

    def fail_to_numpy(_):
        raise AssertionError("dense materialization should stay on the backend")

    try:
        set_backend("jax")
        with monkeypatch.context() as context:
            context.setattr(linear_map_module, "to_numpy", fail_to_numpy)
            pointwise_dense = pointwise.to_matrix()
            selector_dense = selector.to_matrix()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(pointwise_dense).__module__
    assert "jax" in type(selector_dense).__module__
    np.testing.assert_allclose(np.asarray(pointwise_dense), pointwise.to_matrix(backend="numpy"))
    np.testing.assert_allclose(np.asarray(selector_dense), selector.to_matrix(backend="numpy"))


# Composition and materialization


def test_diagonal_composition_avoids_dense_diagonal_materialization():
    """Dense composite materialization scales rows/columns directly."""
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    matrix_map = as_linear_map(matrix)
    left = diagonal_linear_map(np.array([7.0, 11.0]))
    right = diagonal_linear_map(np.array([2.0, 3.0]))

    def fail_dense(_xp):
        raise AssertionError("diagonal dense materialization should not be used")

    object.__setattr__(left, "_dense_array_func", fail_dense)
    object.__setattr__(right, "_dense_array_func", fail_dense)

    np.testing.assert_allclose(
        (left @ matrix_map).to_matrix(backend="numpy"),
        left.diagonal(backend="numpy").reshape(-1, 1) * matrix,
    )
    np.testing.assert_allclose(
        (matrix_map @ right).to_matrix(backend="numpy"),
        matrix * right.diagonal(backend="numpy").reshape(1, -1),
    )


def test_diagonal_composition_preserves_diagonal_metadata():
    """Composed diagonal maps remain diagonal maps."""
    left = diagonal_linear_map(np.array([2.0, 3.0]))
    right = diagonal_linear_map(np.array([5.0, 7.0]))
    composed = left @ right

    np.testing.assert_allclose(composed.diagonal(backend="numpy"), [10.0, 21.0])
    np.testing.assert_allclose(composed.normal_matrix_diag(), [100.0, 441.0])


def test_linear_map_diagonal_accessor_is_strict():
    """Diagonal values are available only for diagonal operators."""
    diagonal_matrix = as_linear_map(np.diag([2.0, 3.0]))
    non_diagonal_matrix = as_linear_map(np.array([[2.0, 1.0], [0.0, 3.0]]))

    np.testing.assert_allclose(diagonal_matrix.diagonal(backend="numpy"), [2.0, 3.0])
    np.testing.assert_allclose(
        (4.0 * diagonal_linear_map(np.array([2.0, 3.0]))).diagonal(backend="numpy"), [8.0, 12.0]
    )
    with pytest.raises(ValueError, match="not diagonal"):
        non_diagonal_matrix.diagonal(backend="numpy")


def test_scaled_dense_map_preserves_composition_structure():
    """Scaled dense maps remain structurally composable."""
    rng = np.random.default_rng(14)
    scalar = -2.5
    left_tensor = rng.normal(size=(2, 3, 4))
    right_tensor = rng.normal(size=(4, 5))
    left = as_linear_map(left_tensor.reshape(6, 4), output_shape=(2, 3), input_shape=(4,))
    right = einsum_linear_map(
        component_tensors=[right_tensor],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(4,),
        input_shape=(5,),
    )

    def fail_dense(_xp):
        raise AssertionError("scaled dense composition should use stored tensor")

    object.__setattr__(left, "_dense_array_func", fail_dense)

    scaled = scalar * left
    composed = scaled @ right

    assert scaled._dense_tensor is not None
    assert composed._einsum_map is not None
    np.testing.assert_allclose(
        composed.to_matrix(backend="numpy"), scalar * left_tensor.reshape(6, 4) @ right_tensor
    )


def test_scaled_einsum_map_preserves_composition_structure():
    """Scaled einsum maps stay structurally composable."""
    rng = np.random.default_rng(15)
    scalar = 1.5 - 0.25j
    left_tensor = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
    right_tensor = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    left = einsum_linear_map(
        component_tensors=[left_tensor],
        einsum_string_dense="abi->abi",
        einsum_string_matvec="abi,i->ab",
        einsum_string_rmatvec="ab,abi->i",
        output_shape=(2, 3),
        input_shape=(4,),
    )
    right = einsum_linear_map(
        component_tensors=[right_tensor],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(4,),
        input_shape=(5,),
    )

    def fail_dense(_xp):
        raise AssertionError("scaled einsum composition should not densify")

    object.__setattr__(left, "_dense_array_func", fail_dense)
    object.__setattr__(right, "_dense_array_func", fail_dense)

    scaled = scalar * left
    composed = scaled @ right
    expected = scalar * left_tensor.reshape(6, 4) @ right_tensor

    assert scaled._einsum_map is not None
    assert composed._einsum_map is not None
    np.testing.assert_allclose(composed.to_matrix(backend="numpy"), expected)
    np.testing.assert_allclose(
        composed.normal_matrix_diag(), np.sum(np.abs(expected) ** 2, axis=0)
    )


def test_to_matrix_reuses_cached_matrix_for_application():
    """A requested explicit matrix is reused for subsequent application."""
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    calls = 0

    def fail_matvec(_):
        raise AssertionError("materialized map should use dense matvec")

    def fail_rmatvec(_):
        raise AssertionError("materialized map should use dense rmatvec")

    def dense_array(xp):
        nonlocal calls
        calls += 1
        return xp.asarray(matrix)

    linear_map = LinearMap(
        shape=matrix.shape,
        dtype=matrix.dtype,
        matvec=fail_matvec,
        rmatvec=fail_rmatvec,
        dense_array=dense_array,
        output_shape=(2,),
        input_shape=(2,),
    )
    np.testing.assert_allclose(linear_map.to_matrix(), matrix)

    x = np.array([7.0, 11.0])
    block = np.eye(2)

    np.testing.assert_allclose(linear_map.matvec(x), matrix @ x)
    np.testing.assert_allclose(linear_map.matmat(block), matrix)
    np.testing.assert_allclose(linear_map.rmatvec(x), matrix.T @ x)
    np.testing.assert_allclose(linear_map.rmatmat(block), matrix.T)
    assert calls == 1


def test_linear_map_to_matrix_materializes_on_requested_backend():
    """to_matrix is the explicit flat materialization API."""
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])

    dense = as_linear_map(matrix).to_matrix(backend="numpy")

    np.testing.assert_allclose(dense, matrix)


def test_wide_matrix_free_map_materializes_through_its_adjoint():
    """Wide maps probe fewer outputs instead of all input columns."""
    matrix = np.arange(30.0).reshape(3, 10)

    def fail_matmat(_):
        raise AssertionError("wide materialization should use the adjoint")

    linear_map = LinearMap(
        shape=matrix.shape,
        dtype=matrix.dtype,
        matvec=lambda values: matrix @ np.asarray(values),
        rmatvec=lambda values: matrix.T @ np.asarray(values),
        matmat=fail_matmat,
        rmatmat=lambda values: matrix.T @ np.asarray(values),
    )

    np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), matrix)


def test_wide_composition_materializes_through_composed_adjoint():
    """Wide compositions do not densify their larger input domain."""
    left_matrix = np.arange(6.0).reshape(2, 3)
    right_matrix = np.arange(30.0).reshape(3, 10)

    def matrix_free(matrix):
        def fail_dense(_xp):
            raise AssertionError("component map should not materialize")

        return LinearMap(
            shape=matrix.shape,
            dtype=matrix.dtype,
            matvec=lambda values: matrix @ np.asarray(values),
            rmatvec=lambda values: matrix.T @ np.asarray(values),
            matmat=lambda values: matrix @ np.asarray(values),
            rmatmat=lambda values: matrix.T @ np.asarray(values),
            dense_array=fail_dense,
        )

    composed = matrix_free(left_matrix) @ matrix_free(right_matrix)

    np.testing.assert_allclose(composed.to_matrix(backend="numpy"), left_matrix @ right_matrix)


def test_linear_map_to_array_returns_shaped_dense_representation():
    """Explicit shaped arrays preserve output/input axis metadata."""
    tensor = np.arange(24.0).reshape(2, 3, 4)
    linear_map = as_linear_map(tensor, output_shape=(2, 3), input_shape=(4,))

    dense_matrix = linear_map.to_matrix(backend="numpy")
    shaped_array = linear_map.to_array()

    assert dense_matrix.shape == (6, 4)
    assert shaped_array.shape == tensor.shape
    assert linear_map.to_matrix(backend="numpy") is dense_matrix
    np.testing.assert_allclose(shaped_array, tensor)
    np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), dense_matrix)
    np.testing.assert_allclose(np.asarray(linear_map.to_array()), shaped_array)
    if get_array_module() is np:
        assert np.shares_memory(linear_map.to_array(), linear_map.to_matrix())


def test_linear_map_adjoint_preserves_shapes_and_diagonal_structure():
    """The explicit adjoint swaps shaped domains without densifying diagonals."""
    matrix = np.array([[1.0 + 2.0j, 3.0], [4.0j, -2.0], [5.0, 6.0 - 1.0j]])
    linear_map = as_linear_map(matrix, output_shape=(3,), input_shape=(2,))
    adjoint = linear_map.adjoint()

    assert adjoint.output_shape == (2,)
    assert adjoint.input_shape == (3,)
    np.testing.assert_allclose(adjoint.to_matrix(), matrix.T.conj())

    diagonal = diagonal_linear_map(np.array([1.0 + 2.0j, -3.0j]))
    assert diagonal.adjoint().is_diagonal
    np.testing.assert_allclose(diagonal.adjoint().diagonal(), [1.0 - 2.0j, 3.0j])


# Backend ownership


@pytest.mark.requires_jax
def test_dense_linear_map_accepts_numpy_inputs_with_jax_backend():
    """Dense maps stay NumPy-facing until called with JAX inputs."""
    previous_backend = jax_enabled()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    x = np.array([0.25, -2.0])

    try:
        set_backend("jax")
        linear_map = as_linear_map(matrix)

        np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), matrix)
        np.testing.assert_allclose(linear_map.matvec(x), matrix @ x)
    finally:
        set_backend(previous_backend)


@pytest.mark.requires_jax
def test_linear_map_matmul_preserves_numpy_operand_backend():
    """``A @ x`` and ``A.matvec(x)`` should have identical backend ownership."""
    previous_backend = jax_enabled()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    vector = np.array([0.25, -2.0])
    block = np.column_stack((vector, vector + 1.0))

    try:
        set_backend("jax")
        linear_map = as_linear_map(matrix)
        vector_result = linear_map @ vector
        block_result = linear_map @ block
    finally:
        set_backend(previous_backend)

    assert isinstance(vector_result, np.ndarray)
    assert isinstance(block_result, np.ndarray)
    np.testing.assert_allclose(vector_result, matrix @ vector)
    np.testing.assert_allclose(block_result, matrix @ block)


@pytest.mark.requires_jax
def test_pointwise_linear_map_accepts_numpy_inputs_with_jax_backend():
    """Pointwise maps stay NumPy-facing until called with JAX inputs."""
    previous_backend = jax_enabled()
    matrix = np.arange(24.0).reshape(2, 3, 4)
    values = np.arange(12.0).reshape(3, 4)
    linear_map = pointwise_matrix_linear_map(matrix)

    try:
        set_backend("jax")
        result = linear_map.matvec(values)
    finally:
        set_backend(previous_backend)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.einsum("abg,bg->ag", matrix, values).reshape(-1))


@pytest.mark.requires_jax
def test_dense_linear_map_preserves_jax_dense_source(monkeypatch):
    """JAX dense inputs should not materialize during creation."""
    import jax.numpy as jnp

    import kompe.math.linear_map as linear_map_module

    matrix = jnp.asarray([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    x = jnp.asarray([0.25, -2.0])

    def fail_asarray(_):
        raise AssertionError("as_linear_map should preserve JAX dense inputs")

    with monkeypatch.context() as context:
        context.setattr(linear_map_module.np, "asarray", fail_asarray)
        linear_map = as_linear_map(matrix)

    result = linear_map.matvec(x)
    with monkeypatch.context() as context:
        context.setattr(linear_map_module, "to_numpy", fail_asarray)
        dense = linear_map.to_matrix(backend="jax")

    assert "jax" in type(result).__module__
    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(np.asarray(result), np.asarray(matrix) @ np.asarray(x))
    np.testing.assert_allclose(np.asarray(dense), np.asarray(matrix))


@pytest.mark.requires_jax
def test_diagonal_linear_map_preserves_jax_dense_source(monkeypatch):
    """JAX diagonal inputs should not materialize during creation."""
    import jax.numpy as jnp

    import kompe.math.linear_map as linear_map_module

    diagonal = jnp.asarray([2.0, 3.0])
    x = jnp.asarray([0.25, -2.0])

    def fail_asarray(_):
        raise AssertionError("as_linear_map should preserve JAX diagonal inputs")

    with monkeypatch.context() as context:
        context.setattr(linear_map_module.np, "asarray", fail_asarray)
        linear_map = as_linear_map(diagonal)

    result = linear_map.matvec(x)
    with monkeypatch.context() as context:
        context.setattr(linear_map_module, "to_numpy", fail_asarray)
        dense = linear_map.to_matrix(backend="jax")

    assert "jax" in type(result).__module__
    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(np.asarray(result), np.asarray(diagonal) * np.asarray(x))
    np.testing.assert_allclose(np.asarray(dense), np.diag(np.asarray(diagonal)))


@pytest.mark.requires_jax
def test_linear_map_dense_uses_active_backend():
    """Dense materialization can stay on the active backend."""
    previous_backend = jax_enabled()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    base = as_linear_map(matrix)
    matrix_free = LinearMap(
        shape=base.shape,
        dtype=base.dtype,
        matvec=base.matvec,
        rmatvec=base.rmatvec,
        matmat=base.matmat,
        rmatmat=base.rmatmat,
    )

    try:
        set_backend("jax")
        dense = matrix_free.to_matrix()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(dense, matrix)


@pytest.mark.requires_jax
def test_linear_map_backend_operands_drives_matrix_free_batches():
    """Matrix-free batching follows operator backend context."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    backend_operands = jnp.asarray(0.0)

    def matvec(vec):
        xp = get_array_module(vec, backend_operands)
        return xp.asarray(matrix) @ xp.asarray(vec)

    def rmatvec(vec):
        xp = get_array_module(vec, backend_operands)
        return xp.asarray(matrix).T @ xp.asarray(vec)

    linear_map = LinearMap(
        shape=matrix.shape,
        dtype=matrix.dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        backend_operands=(backend_operands,),
    )

    try:
        set_backend("numpy")
        result = linear_map.matmat(np.eye(2))
        adjoint_result = linear_map.rmatmat(np.eye(2))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    assert "jax" in type(adjoint_result).__module__
    np.testing.assert_allclose(np.asarray(result), matrix)
    np.testing.assert_allclose(np.asarray(adjoint_result), matrix.T)


# Sparse maps


def test_sparse_linear_map_uses_sparse_normal_diagonal():
    """Sparse maps avoid generic densifying for normal diagonals."""
    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    linear_map = as_linear_map(csr_matrix(matrix))
    x = np.array([0.25, -2.0])
    y = np.array([1.0, 0.5, -1.0])

    np.testing.assert_allclose(linear_map.normal_matrix_diag(), np.sum(matrix**2, axis=0))
    np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), matrix)
    np.testing.assert_allclose(linear_map.matvec(x[:, None]), matrix @ x)
    np.testing.assert_allclose(linear_map.rmatvec(y[:, None]), matrix.T @ y)


@pytest.mark.requires_jax
def test_scipy_sparse_linear_map_stays_numpy_facing_with_jax_backend():
    """SciPy sparse maps select a device from their input, not construction state."""
    previous_backend = jax_enabled()
    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    x = np.array([0.25, -2.0])

    try:
        set_backend("jax")
        linear_map = as_linear_map(csr_matrix(matrix))
        result = linear_map.matvec(x)
    finally:
        set_backend(previous_backend)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, matrix @ x)


@pytest.mark.requires_jax
def test_jax_sparse_linear_map_uses_sparse_normal_diagonal():
    """JAX sparse maps preserve complex adjoints and sparse normal diagonals."""
    from jax.experimental.sparse import BCOO

    matrix = np.array([[2.0 + 1.0j, 0.0], [0.0, 3.0 - 2.0j], [1.0, -1.0j]])
    linear_map = as_linear_map(BCOO.from_scipy_sparse(csr_matrix(matrix)))
    y = np.array([1.0j, 0.5, -1.0 + 2.0j])

    np.testing.assert_allclose(linear_map.rmatvec(y), matrix.conj().T @ y)

    def fail_matmat(_block):
        raise AssertionError("normal_matrix_diag should use sparse metadata")

    def fail_dense(_xp):
        raise AssertionError("normal_matrix_diag should not densify")

    object.__setattr__(linear_map, "_matmat", fail_matmat)
    object.__setattr__(linear_map, "_dense_array_func", fail_dense)

    np.testing.assert_allclose(
        linear_map.normal_matrix_diag(), np.sum(np.abs(matrix) ** 2, axis=0)
    )


@pytest.mark.requires_jax
def test_sparse_linear_map_preserves_structure_for_explicit_jax_inputs(monkeypatch):
    """SciPy sparse maps transfer sparse structure for JAX operands."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    matrix = np.array([[2.0 + 1.0j, 0.0], [0.0, 3.0 - 2.0j], [1.0, -1.0j]], dtype=np.complex64)
    x = np.array([0.25 - 0.5j, -2.0 + 0.75j], dtype=np.complex64)
    y = np.array([1.0j, 0.5, -1.0 + 2.0j], dtype=np.complex64)

    try:
        set_backend("numpy")
        linear_map = as_linear_map(csr_matrix(matrix))

        def fail_toarray(*_args, **_kwargs):
            raise AssertionError("JAX sparse application should not densify")

        monkeypatch.setattr(csr_matrix, "toarray", fail_toarray)
        result = linear_map.matvec(jnp.asarray(x))
        adjoint_result = linear_map.rmatvec(jnp.asarray(y))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(result).__module__
    assert "jax" in type(adjoint_result).__module__
    assert linear_map._dense_cache == {}
    np.testing.assert_allclose(np.asarray(result), matrix @ x)
    np.testing.assert_allclose(np.asarray(adjoint_result), matrix.T.conj() @ y)


def test_composed_linear_map_normal_diagonal_uses_matmat_path():
    """Composed maps do not densify for normal diagonals."""
    matrix = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    weights = np.array([2.0, -1.0, 0.25])
    base = as_linear_map(matrix)

    def fail_dense_array(_):
        raise AssertionError("normal_matrix_diag should not call dense")

    matrix_free = LinearMap(
        shape=base.shape,
        dtype=base.dtype,
        matvec=base.matvec,
        rmatvec=base.rmatvec,
        matmat=base.matmat,
        rmatmat=base.rmatmat,
        dense_array=fail_dense_array,
    )
    composed = diagonal_linear_map(weights) @ matrix_free
    expected = np.sum((weights[:, None] * matrix) ** 2, axis=0)

    np.testing.assert_allclose(composed.normal_matrix_diag(), expected)


# Tensor and einsum maps


def test_einsum_linear_map_matches_matrix_operations():
    """Einsum contractions can back the LinearMap interface."""
    matrix = np.array([[1.0, 2.0, -1.0], [0.0, 3.0, 4.0]])
    linear_map = einsum_linear_map(
        component_tensors=[matrix],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(3,),
    )
    x = np.array([2.0, -1.0, 0.5])

    np.testing.assert_allclose(linear_map.matvec(x), matrix @ x)
    np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), matrix)
    assert linear_map.output_shape == (2,)
    assert linear_map.input_shape == (3,)


def test_einsum_linear_map_batched_application_matches_dense():
    """Einsum-backed batched application matches dense products."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 6))
    b = rng.normal(size=(6, 4))
    linear_map = einsum_linear_map(
        component_tensors=[a, b],
        einsum_string_dense="ij,jk->ik",
        einsum_string_matvec="ij,jk,k->i",
        einsum_string_rmatvec="i,ij,jk->k",
        output_shape=(5,),
        input_shape=(4,),
    )
    dense = linear_map.to_matrix(backend="numpy")
    x_block = rng.normal(size=(4, 7))
    y_block = rng.normal(size=(5, 7))

    np.testing.assert_allclose(linear_map.matmat(x_block), dense @ x_block)
    np.testing.assert_allclose(linear_map.rmatmat(y_block), dense.T @ y_block)
    np.testing.assert_allclose(linear_map.normal_matrix_diag(), np.sum(dense**2, axis=0))


def test_einsum_linear_map_composition_fuses_exact_shapes():
    """Exact shaped einsum compositions stay einsum-backed."""
    rng = np.random.default_rng(3)
    left_tensor = rng.normal(size=(2, 3, 4, 5)) + 1j * rng.normal(size=(2, 3, 4, 5))
    right_tensor = rng.normal(size=(4, 5, 6)) + 1j * rng.normal(size=(4, 5, 6))
    left = einsum_linear_map(
        component_tensors=[left_tensor],
        einsum_string_dense="abij->abij",
        einsum_string_matvec="abij,ij->ab",
        einsum_string_rmatvec="ab,abij->ij",
        output_shape=(2, 3),
        input_shape=(4, 5),
    )
    right = einsum_linear_map(
        component_tensors=[right_tensor],
        einsum_string_dense="ijk->ijk",
        einsum_string_matvec="ijk,k->ij",
        einsum_string_rmatvec="ij,ijk->k",
        output_shape=(4, 5),
        input_shape=(6,),
    )
    expected_dense = left_tensor.reshape(6, 20) @ right_tensor.reshape(20, 6)

    def fail_dense(_xp):
        raise AssertionError("fused einsum composition should not densify inputs")

    object.__setattr__(left, "_dense_array_func", fail_dense)
    object.__setattr__(right, "_dense_array_func", fail_dense)

    composed = left @ right
    x = rng.normal(size=6) + 1j * rng.normal(size=6)
    y = rng.normal(size=6) + 1j * rng.normal(size=6)
    block = rng.normal(size=(6, 4)) + 1j * rng.normal(size=(6, 4))

    assert composed._einsum_map is not None

    def fail_probe():
        raise AssertionError("fused normal diagonal should use direct einsum")

    object.__setattr__(composed._einsum_map, "_normal_matrix_diag_probe", fail_probe)

    np.testing.assert_allclose(composed.to_matrix(backend="numpy"), expected_dense)
    np.testing.assert_allclose(composed.matvec(x), expected_dense @ x)
    np.testing.assert_allclose(composed.matmat(block), expected_dense @ block)
    np.testing.assert_allclose(composed.rmatvec(y), expected_dense.conj().T @ y)
    np.testing.assert_allclose(
        composed.normal_matrix_diag(), np.sum(np.abs(expected_dense) ** 2, axis=0)
    )


def test_dense_and_einsum_linear_map_composition_fuses_exact_shapes():
    """Dense maps can fuse with einsum maps."""
    rng = np.random.default_rng(5)
    left_tensor = rng.normal(size=(2, 3, 4, 5)) + 1j * rng.normal(size=(2, 3, 4, 5))
    right_tensor = rng.normal(size=(4, 5, 6)) + 1j * rng.normal(size=(4, 5, 6))
    left = as_linear_map(left_tensor.reshape(6, 20), output_shape=(2, 3), input_shape=(4, 5))
    right = einsum_linear_map(
        component_tensors=[right_tensor],
        einsum_string_dense="ijk->ijk",
        einsum_string_matvec="ijk,k->ij",
        einsum_string_rmatvec="ij,ijk->k",
        output_shape=(4, 5),
        input_shape=(6,),
    )
    expected_dense = left_tensor.reshape(6, 20) @ right_tensor.reshape(20, 6)

    def fail_dense(_xp):
        raise AssertionError("fused composition should not densify inputs")

    object.__setattr__(left, "_dense_array_func", fail_dense)
    object.__setattr__(right, "_dense_array_func", fail_dense)

    composed = left @ right
    x = rng.normal(size=6) + 1j * rng.normal(size=6)
    y = rng.normal(size=6) + 1j * rng.normal(size=6)

    assert composed._einsum_map is not None
    np.testing.assert_allclose(composed.to_matrix(backend="numpy"), expected_dense)
    np.testing.assert_allclose(composed.matvec(x), expected_dense @ x)
    np.testing.assert_allclose(composed.rmatvec(y), expected_dense.conj().T @ y)


def test_einsum_and_dense_linear_map_composition_fuses_exact_shapes():
    """Einsum maps can fuse with dense maps on the right."""
    rng = np.random.default_rng(6)
    left_tensor = rng.normal(size=(2, 3, 4, 5)) + 1j * rng.normal(size=(2, 3, 4, 5))
    right_tensor = rng.normal(size=(4, 5, 6)) + 1j * rng.normal(size=(4, 5, 6))
    left = einsum_linear_map(
        component_tensors=[left_tensor],
        einsum_string_dense="abij->abij",
        einsum_string_matvec="abij,ij->ab",
        einsum_string_rmatvec="ab,abij->ij",
        output_shape=(2, 3),
        input_shape=(4, 5),
    )
    right = as_linear_map(right_tensor.reshape(20, 6), output_shape=(4, 5), input_shape=(6,))
    expected_dense = left_tensor.reshape(6, 20) @ right_tensor.reshape(20, 6)

    def fail_dense(_xp):
        raise AssertionError("fused composition should not densify inputs")

    object.__setattr__(left, "_dense_array_func", fail_dense)
    object.__setattr__(right, "_dense_array_func", fail_dense)

    composed = left @ right
    x = rng.normal(size=6) + 1j * rng.normal(size=6)
    y = rng.normal(size=6) + 1j * rng.normal(size=6)

    assert composed._einsum_map is not None
    np.testing.assert_allclose(composed.to_matrix(backend="numpy"), expected_dense)
    np.testing.assert_allclose(composed.matvec(x), expected_dense @ x)
    np.testing.assert_allclose(composed.rmatvec(y), expected_dense.conj().T @ y)


def test_dense_linear_map_composition_fuses_exact_shapes():
    """Dense-dense products use einsum-backed composition."""
    rng = np.random.default_rng(7)
    left = as_linear_map(rng.normal(size=(4, 3)))
    right = as_linear_map(rng.normal(size=(3, 2)))

    composed = left @ right

    assert composed._einsum_map is not None
    np.testing.assert_allclose(
        composed.to_matrix(backend="numpy"),
        left.to_matrix(backend="numpy") @ right.to_matrix(backend="numpy"),
    )


def test_diagonal_and_einsum_composition_fuses_exact_shapes():
    """Shaped diagonals fuse as elementwise einsum factors."""
    rng = np.random.default_rng(11)
    left_diag_values = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    right_diag_values = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    tensor = rng.normal(size=(2, 3, 4, 5)) + 1j * rng.normal(size=(2, 3, 4, 5))
    base = einsum_linear_map(
        component_tensors=[tensor],
        einsum_string_dense="abij->abij",
        einsum_string_matvec="abij,ij->ab",
        einsum_string_rmatvec="ab,abij->ij",
        output_shape=(2, 3),
        input_shape=(4, 5),
    )
    left_diag = diagonal_linear_map(
        left_diag_values.reshape(-1), output_shape=(2, 3), input_shape=(2, 3)
    )
    right_diag = diagonal_linear_map(
        right_diag_values.reshape(-1), output_shape=(4, 5), input_shape=(4, 5)
    )
    expected_left = left_diag_values.reshape(6, 1) * tensor.reshape(6, 20)
    expected_right = tensor.reshape(6, 20) * right_diag_values.reshape(1, 20)

    def fail_dense(_xp):
        raise AssertionError("diagonal fusion should not materialize dense diagonal")

    object.__setattr__(left_diag, "_dense_array_func", fail_dense)
    object.__setattr__(right_diag, "_dense_array_func", fail_dense)

    left_composed = left_diag @ base
    right_composed = base @ right_diag
    x = rng.normal(size=20) + 1j * rng.normal(size=20)
    y = rng.normal(size=6) + 1j * rng.normal(size=6)

    assert left_composed._einsum_map is not None
    assert right_composed._einsum_map is not None
    np.testing.assert_allclose(left_composed.to_matrix(backend="numpy"), expected_left)
    np.testing.assert_allclose(right_composed.to_matrix(backend="numpy"), expected_right)
    np.testing.assert_allclose(left_composed.matvec(x), expected_left @ x)
    np.testing.assert_allclose(right_composed.rmatvec(y), expected_right.conj().T @ y)


def test_diagonal_composition_falls_back_for_flat_only_shape_match():
    """Flat-only diagonal shape matches stay on generic composition."""
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(6, 4))
    diagonal = diagonal_linear_map(rng.normal(size=6), output_shape=(6,), input_shape=(6,))
    shaped_matrix = as_linear_map(matrix, output_shape=(2, 3), input_shape=(4,))

    composed = diagonal @ shaped_matrix

    assert composed._einsum_map is None
    np.testing.assert_allclose(composed.to_matrix(backend="numpy"), diagonal.to_matrix() @ matrix)


def test_dense_chain_then_einsum_composition_fuses_later():
    """Dense-only chains remain available for later fusion."""
    rng = np.random.default_rng(8)
    left_tensor = rng.normal(size=(2, 3, 4))
    middle_tensor = rng.normal(size=(4, 5))
    right_tensor = rng.normal(size=(5, 6))
    left = as_linear_map(left_tensor.reshape(6, 4), output_shape=(2, 3), input_shape=(4,))
    middle = as_linear_map(middle_tensor, output_shape=(4,), input_shape=(5,))
    right = einsum_linear_map(
        component_tensors=[right_tensor],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(5,),
        input_shape=(6,),
    )
    dense_chain = left @ middle
    expected_dense = left_tensor.reshape(6, 4) @ middle_tensor @ right_tensor

    assert dense_chain._einsum_map is not None
    composed = dense_chain @ right

    assert composed._einsum_map is not None
    np.testing.assert_allclose(composed.to_matrix(backend="numpy"), expected_dense)


def test_einsum_then_dense_chain_composition_fuses_later():
    """Einsum maps can fuse through a precomposed dense chain."""
    rng = np.random.default_rng(9)
    left_tensor = rng.normal(size=(2, 3, 4))
    middle_tensor = rng.normal(size=(4, 5))
    right_tensor = rng.normal(size=(5, 6))
    left = einsum_linear_map(
        component_tensors=[left_tensor],
        einsum_string_dense="abi->abi",
        einsum_string_matvec="abi,i->ab",
        einsum_string_rmatvec="ab,abi->i",
        output_shape=(2, 3),
        input_shape=(4,),
    )
    middle = as_linear_map(middle_tensor, output_shape=(4,), input_shape=(5,))
    right = as_linear_map(right_tensor, output_shape=(5,), input_shape=(6,))
    dense_chain = middle @ right
    expected_dense = left_tensor.reshape(6, 4) @ middle_tensor @ right_tensor

    assert dense_chain._einsum_map is not None
    composed = left @ dense_chain

    assert composed._einsum_map is not None
    np.testing.assert_allclose(composed.to_matrix(backend="numpy"), expected_dense)


def test_four_factor_structured_chains_fuse_for_all_groupings():
    """Four-factor structured chains fuse for all groupings."""
    rng = np.random.default_rng(10)
    shapes = [((2, 3), (4,)), ((4,), (4,)), ((4,), (5,)), ((5,), (5,))]
    matrices = [
        rng.normal(size=(int(np.prod(output_shape)), int(np.prod(input_shape))))
        for output_shape, input_shape in shapes
    ]

    def make_dense(index):
        output_shape, input_shape = shapes[index]
        matrix = matrices[index]
        return (matrix, as_linear_map(matrix, output_shape=output_shape, input_shape=input_shape))

    def make_einsum(index):
        output_shape, input_shape = shapes[index]
        matrix = matrices[index]
        tensor = matrix.reshape(output_shape + input_shape)
        labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        output_labels = labels[: len(output_shape)]
        input_labels = labels[len(output_shape) : len(output_shape) + len(input_shape)]
        all_labels = output_labels + input_labels
        return (
            matrix,
            einsum_linear_map(
                component_tensors=[tensor],
                einsum_string_dense=f"{all_labels}->{all_labels}",
                einsum_string_matvec=(f"{all_labels},{input_labels}->{output_labels}"),
                einsum_string_rmatvec=(f"{output_labels},{all_labels}->{input_labels}"),
                output_shape=output_shape,
                input_shape=input_shape,
            ),
        )

    def make_diagonal(index):
        output_shape, input_shape = shapes[index]
        if output_shape != input_shape:
            return make_dense(index)
        diagonal = rng.normal(size=int(np.prod(output_shape)))
        return (
            np.diag(diagonal),
            diagonal_linear_map(diagonal, output_shape=output_shape, input_shape=input_shape),
        )

    def make_factor(kind, index):
        if kind == "E":
            return make_einsum(index)
        if kind == "G":
            return make_diagonal(index)
        return make_dense(index)

    def grouped(maps, grouping):
        first, second, third, fourth = maps
        if grouping == 0:
            return ((first @ second) @ third) @ fourth
        if grouping == 1:
            return (first @ (second @ third)) @ fourth
        if grouping == 2:
            return (first @ second) @ (third @ fourth)
        if grouping == 3:
            return first @ ((second @ third) @ fourth)
        return first @ (second @ (third @ fourth))

    for kinds in product("DEG", repeat=4):
        matrices_for_kinds = []
        maps = []
        for index, kind in enumerate(kinds):
            matrix, linear_map = make_factor(kind, index)
            matrices_for_kinds.append(matrix)
            maps.append(linear_map)
        expected_dense = (
            matrices_for_kinds[0]
            @ matrices_for_kinds[1]
            @ matrices_for_kinds[2]
            @ matrices_for_kinds[3]
        )
        for grouping in range(5):
            composed = grouped(maps, grouping)
            assert composed._einsum_map is not None
            np.testing.assert_allclose(composed.to_matrix(backend="numpy"), expected_dense)


def test_four_factor_structured_chain_regroups_independently():
    """Grouping does not affect four-factor structured chains."""
    rng = np.random.default_rng(13)
    shapes = [((2, 3), (4,)), ((4,), (4,)), ((4,), (5,)), ((5,), (5,))]
    factors = []
    matrices = []
    for index, (output_shape, input_shape) in enumerate(shapes):
        matrix = rng.normal(size=(int(np.prod(output_shape)), int(np.prod(input_shape))))
        if index in (1, 3):
            diagonal = rng.normal(size=int(np.prod(output_shape)))
            matrix = np.diag(diagonal)
            linear_map = diagonal_linear_map(
                diagonal, output_shape=output_shape, input_shape=input_shape
            )
        elif index == 2:
            tensor = matrix.reshape(output_shape + input_shape)
            linear_map = einsum_linear_map(
                component_tensors=[tensor],
                einsum_string_dense="ij->ij",
                einsum_string_matvec="ij,j->i",
                einsum_string_rmatvec="i,ij->j",
                output_shape=output_shape,
                input_shape=input_shape,
            )
        else:
            linear_map = as_linear_map(matrix, output_shape=output_shape, input_shape=input_shape)
        matrices.append(matrix)
        factors.append(linear_map)

    expected_dense = matrices[0] @ matrices[1] @ matrices[2] @ matrices[3]
    groupings = [
        ((factors[0] @ factors[1]) @ factors[2]) @ factors[3],
        (factors[0] @ (factors[1] @ factors[2])) @ factors[3],
        (factors[0] @ factors[1]) @ (factors[2] @ factors[3]),
        factors[0] @ ((factors[1] @ factors[2]) @ factors[3]),
        factors[0] @ (factors[1] @ (factors[2] @ factors[3])),
    ]

    for composed in groupings:
        assert composed._einsum_map is not None
        np.testing.assert_allclose(composed.to_matrix(backend="numpy"), expected_dense)


def test_einsum_linear_map_composition_falls_back_for_flat_only_match():
    """Flat-compatible compositions do not fuse for different axes."""
    rng = np.random.default_rng(4)
    left_tensor = rng.normal(size=(2, 2, 3, 2))
    right_tensor = rng.normal(size=(6, 5))
    left = einsum_linear_map(
        component_tensors=[left_tensor],
        einsum_string_dense="abij->abij",
        einsum_string_matvec="abij,ij->ab",
        einsum_string_rmatvec="ab,abij->ij",
        output_shape=(2, 2),
        input_shape=(3, 2),
    )
    right = einsum_linear_map(
        component_tensors=[right_tensor],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(6,),
        input_shape=(5,),
    )

    composed = left @ right

    assert composed._einsum_map is None
    np.testing.assert_allclose(
        composed.to_matrix(backend="numpy"),
        left.to_matrix(backend="numpy") @ right.to_matrix(backend="numpy"),
    )


def test_einsum_linear_map_from_matvec_derives_adjoint_and_dense():
    """Forward-only einsum maps derive dense and adjoint forms."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 4))
    b = rng.normal(size=(4, 2))
    linear_map = einsum_linear_map_from_matvec(
        component_tensors=[a, b],
        einsum_string_matvec="ij,jk,k->i",
        output_shape=(3,),
        input_shape=(2,),
    )
    dense = a @ b
    x = rng.normal(size=2)
    y = rng.normal(size=3)

    np.testing.assert_allclose(linear_map.to_matrix(backend="numpy"), dense)
    np.testing.assert_allclose(linear_map.matvec(x), dense @ x)
    np.testing.assert_allclose(linear_map.rmatvec(y), dense.T @ y)


def test_einsum_linear_map_from_matvec_rejects_ambiguous_labels():
    """Derived dense maps require separate input and output labels."""
    with pytest.raises(ValueError, match="distinct labels"):
        einsum_linear_map_from_matvec(
            component_tensors=[np.ones(3)],
            einsum_string_matvec="i,i->i",
            output_shape=(3,),
            input_shape=(3,),
        )


@pytest.mark.requires_jax
def test_einsum_linear_map_dense_materialization_uses_active_backend():
    """Einsum-backed dense materialization can stay on JAX."""
    previous_backend = jax_enabled()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    linear_map = einsum_linear_map(
        component_tensors=[matrix],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(2,),
    )

    try:
        set_backend("jax")
        dense = linear_map.to_matrix()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(np.asarray(dense), matrix)


@pytest.mark.requires_jax
def test_einsum_linear_map_dense_uses_active_backend():
    """Einsum-backed LinearMap preserves the active array backend."""
    previous_backend = jax_enabled()
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    linear_map = einsum_linear_map(
        component_tensors=[matrix],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(2,),
    )

    try:
        set_backend("jax")
        dense = linear_map.to_matrix()
    finally:
        set_backend(previous_backend)

    assert "jax" in type(dense).__module__
    np.testing.assert_allclose(np.asarray(dense), matrix)


@pytest.mark.requires_jax
def test_einsum_linear_map_dtype_does_not_materialize_jax_components(monkeypatch):
    """Einsum-backed dtype should only inspect dtype metadata."""
    import jax.numpy as jnp

    import kompe.math.einsum as einsum_map_module

    linear_map = einsum_linear_map(
        component_tensors=[jnp.asarray([[1.0, 2.0], [3.0, 5.0]])],
        einsum_string_dense="ij->ij",
        einsum_string_matvec="ij,j->i",
        einsum_string_rmatvec="i,ij->j",
        output_shape=(2,),
        input_shape=(2,),
    )

    def fail_to_numpy(_):
        raise AssertionError("dtype should not materialize component arrays")

    monkeypatch.setattr(einsum_map_module, "to_numpy", fail_to_numpy)

    assert linear_map.dtype == np.dtype(jnp.asarray(0.0).dtype)


def test_einsum_linear_map_complex_adjoint_matches_dense():
    """Einsum adjoints match dense conjugate transpose products."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    b = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    linear_map = einsum_linear_map(
        component_tensors=[a, b],
        einsum_string_dense="ij,jk->ik",
        einsum_string_matvec="ij,jk,k->i",
        einsum_string_rmatvec="i,ij,jk->k",
        output_shape=(3,),
        input_shape=(2,),
    )
    dense = linear_map.to_matrix(backend="numpy")
    y = rng.normal(size=3) + 1j * rng.normal(size=3)
    y_block = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))

    np.testing.assert_allclose(linear_map.rmatvec(y), dense.conj().T @ y)
    np.testing.assert_allclose(linear_map.rmatmat(y_block), dense.conj().T @ y_block)
    np.testing.assert_allclose(linear_map.normal_matrix_diag(), np.sum(np.abs(dense) ** 2, axis=0))


def test_einsum_complex_normal_diagonal_probe_is_real():
    """The general probe retains complex work values but returns a real norm."""
    diagonal = np.array([1.0 + 2.0j, -3.0 + 0.5j])
    linear_map = einsum_linear_map(
        component_tensors=[diagonal],
        einsum_string_dense="i->ii",
        einsum_string_matvec="i,i->i",
        einsum_string_rmatvec="i,i->i",
        output_shape=(2,),
        input_shape=(2,),
    )

    result = linear_map.normal_matrix_diag()

    assert not np.issubdtype(result.dtype, np.complexfloating)
    np.testing.assert_allclose(result, np.abs(diagonal) ** 2)


def test_complex_identity_normal_diagonal_is_real():
    """The normal diagonal is real even when the map itself is complex."""
    result = identity_linear_map(3, dtype=np.complex128).normal_matrix_diag()

    assert not np.issubdtype(result.dtype, np.complexfloating)
    np.testing.assert_array_equal(result, np.ones(3))


# Least-squares integration


def test_least_squares_accepts_linear_map_and_sparse_inputs():
    """LeastSquaresProblem normalizes LinearMap and sparse operators."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]

    for operator in [as_linear_map(A), csr_matrix(A)]:
        problem = LeastSquaresProblem(A=operator, solution_shape=2, data_shapes=3)
        solver = LeastSquaresSolver(solver="lsmr", tolerance=1e-12)
        solution = solver.solve(problem, rhs, maxiter=200)
        np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)
