"""Least-squares problem definition."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cached_property
from typing import Any, TypeAlias

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator

from kompe.cache import BoundedCache
from kompe.math.backend import get_array_module, synchronize_linalg_result, to_numpy
from kompe.math.linear_map import LinearMap, as_linear_map, vstack_linear_maps

OperatorInput: TypeAlias = np.ndarray | scipy.sparse.spmatrix | LinearOperator | LinearMap
OperatorInputList: TypeAlias = OperatorInput | list[OperatorInput]
NumericInputList: TypeAlias = float | list[float]
_NORMAL_PINV_CACHE_VERSION = 1


class LeastSquaresProblem:
    """Data fit and optional regularization in one coefficient space.

    With one data term and one regularization term, the solved objective is

    ``||W (A x - b)||² + ||s L x||²``.

    ``sqrt_weights`` supplies the diagonal of ``W`` (so its square gives the
    statistical or area weight). ``regularization_strengths`` supplies the
    dimensionless relative strength before Kompe balances the median non-zero
    diagonals of ``A* A`` and ``L* L``. ``s`` is the resulting row scale: the
    square root of that strength times the balancing factor. The same rule is
    applied term by term when several data or regularization operators are
    provided.
    """

    def __init__(
        self,
        A: OperatorInputList,
        solution_shape: int | tuple[int, ...],
        data_shapes: Any | list[Any],
        sqrt_weights: Any | list[Any] | None = None,
        regularization_strengths: NumericInputList | None = None,
        regularization_operators: OperatorInputList | None = None,
        operator_cache: Any | None = None,
        cache_identity: Any | None = None,
        data_normal_matrix_builder: Callable[[], Any] | None = None,
    ):
        self.solution_shape = (
            (solution_shape,) if isinstance(solution_shape, int) else tuple(solution_shape)
        )
        self.solution_size = math.prod(self.solution_shape)
        self._dense_normal_equation_cache: dict[Any, tuple[Any, Any]] = {}
        self._dense_normal_pinv_cache = BoundedCache(2)
        self.operator_cache = operator_cache
        self.cache_identity = cache_identity
        self.data_normal_matrix_builder = data_normal_matrix_builder
        self._data_normal_matrix_cache = None

        self._process_data_terms(A, data_shapes, sqrt_weights)
        self._process_regularization_terms(regularization_operators, regularization_strengths)

    def _process_data_terms(self, A_in, data_shapes_in, sqrt_weights_in):
        A_list = self._prepare_input_list(A_in, "A")
        self.num_data_terms = len(A_list)
        self.data_shapes = self._normalize_data_shapes(data_shapes_in, self.num_data_terms)
        self.A = [
            as_linear_map(op, output_shape=self.data_shapes[i], input_shape=self.solution_shape)
            for i, op in enumerate(A_list)
        ]
        sqrt_weights_list = self._prepare_input_list(
            sqrt_weights_in, "sqrt_weights", count=self.num_data_terms
        )
        self.sqrt_weights = [
            self._create_weight_operator(w, self.data_shapes[i])
            for i, w in enumerate(sqrt_weights_list)
        ]

    def _process_regularization_terms(self, regularization_operators, regularization_strengths):
        operators = self._prepare_input_list(
            regularization_operators, "regularization_operators", is_optional=True
        )
        self.num_reg_terms = len(operators)
        self.regularization_operators = [
            as_linear_map(operator, input_shape=self.solution_shape)
            if operator is not None
            else None
            for operator in operators
        ]
        self.regularization_strengths = self._validate_regularization_strengths(
            self._prepare_input_list(
                regularization_strengths,
                "regularization_strengths",
                count=self.num_reg_terms,
                default_val=0.0,
            )
        )

    def _create_weight_operator(self, w_val: Any, shape: tuple[int, ...]) -> LinearMap | None:
        if w_val is None:
            return None
        flat_dim = math.prod(shape)
        if not isinstance(w_val, (LinearMap, LinearOperator)) and not scipy.sparse.issparse(w_val):
            arr_shape = getattr(w_val, "shape", None)
            if arr_shape is None:
                arr_shape = np.shape(w_val)
            if arr_shape is not None and tuple(arr_shape) == shape:
                xp = get_array_module(w_val)
                w_val = xp.asarray(w_val).reshape(flat_dim)
        return as_linear_map(w_val, output_shape=shape, input_shape=shape)

    @cached_property
    def regularization_row_scales(self) -> list[float]:
        """Return row scales for the balanced regularization operators."""
        if not any(
            operator is not None and self.regularization_strengths[index] > 0.0
            for index, operator in enumerate(self.regularization_operators)
        ):
            return [0.0] * len(self.regularization_operators)
        if self.data_normal_matrix_builder is None:
            diag_A_T_A = self.data_operator.normal_matrix_diag()
        else:
            diag_A_T_A = np.diag(self._custom_data_normal_matrix()).real
        active_diag_A = diag_A_T_A[diag_A_T_A > 0]
        data_term_scale = np.median(active_diag_A) if active_diag_A.size > 0 else 1.0
        regularization_row_scales = []
        for i, L_item in enumerate(self.regularization_operators):
            strength = self.regularization_strengths[i]
            if strength == 0 or L_item is None:
                regularization_row_scales.append(0.0)
                continue
            diag_L_T_L = L_item.normal_matrix_diag()
            active_diag_L = diag_L_T_L[diag_L_T_L > 0]
            if active_diag_L.size == 0:
                raise ValueError(
                    f"regularization_operators[{i}] is zero but has positive strength."
                )
            reg_term_scale = np.median(active_diag_L)
            scale_factor = math.sqrt(data_term_scale / reg_term_scale)
            regularization_row_scales.append(math.sqrt(strength) * scale_factor)
        return regularization_row_scales

    @cached_property
    def data_operator(self) -> LinearMap:
        """Assemble the data operator without regularization."""
        row_maps = [
            a_item if self.sqrt_weights[i] is None else self.sqrt_weights[i] @ a_item
            for i, a_item in enumerate(self.A)
        ]
        return vstack_linear_maps(row_maps, input_shape=self.solution_shape)

    def system_matrix(self, *, backend=None) -> Any:
        """Materialize the regularized system on the requested backend."""
        return self.system_operator.to_matrix(backend=backend)

    def dense_normal_equations(self) -> tuple[Any, Any, Any, Any]:
        """Return dense system, adjoint, and normal matrix."""
        system_matrix = self.system_matrix()
        xp = get_array_module(system_matrix)
        if xp not in self._dense_normal_equation_cache:
            system_adjoint = system_matrix.T.conj()
            normal_matrix = system_adjoint @ system_matrix
            self._dense_normal_equation_cache[xp] = (system_adjoint, normal_matrix)
        else:
            system_adjoint, normal_matrix = self._dense_normal_equation_cache[xp]
        return xp, system_matrix, system_adjoint, normal_matrix

    def dense_normal_pinv(self, tolerance: float) -> Any:
        """Return cached pseudo-inverse of the dense normal matrix."""
        xp = get_array_module()
        key = (xp, float(tolerance))

        def compute():
            normal_matrix = self.dense_normal_matrix()
            normal_pinv = synchronize_linalg_result(
                xp.linalg.pinv(normal_matrix, rtol=tolerance, hermitian=True)
            )
            # Repeated solves need the pseudo-inverse and the lazy
            # data map, not a separate dense regularized system or
            # normal matrix used to construct it. Keep a materialized
            # data map because the following adjoint products reuse it.
            self._dense_normal_equation_cache.clear()
            regularized_system = self.system_operator
            if regularized_system is not self.data_operator:
                regularized_system.clear_dense_cache()
            self._data_normal_matrix_cache = None
            return normal_pinv

        def build():
            if self.operator_cache is None or self.cache_identity is None:
                return compute()

            cached = self.operator_cache.get_or_create(
                "least_squares_normal_pinv",
                {
                    "algorithm": "least_squares_normal_pinv",
                    "version": _NORMAL_PINV_CACHE_VERSION,
                    "problem": self.cache_identity,
                    "backend": xp.__name__,
                    "tolerance": float(tolerance),
                },
                lambda: to_numpy(compute()),
            )
            return xp.asarray(cached)

        return self._dense_normal_pinv_cache.get_or_create(key, build)

    def dense_normal_matrix(self) -> Any:
        """Return the regularized normal matrix."""
        if self.data_normal_matrix_builder is None:
            return self.dense_normal_equations()[3]

        data_normal = self._custom_data_normal_matrix()
        regularization_terms = self._active_regularization_terms()
        self._data_normal_matrix_cache = None
        if not regularization_terms:
            return get_array_module().asarray(data_normal)

        # The builder may return a shared or cached matrix. Regularization
        # belongs to this problem and must not alter the builder's data.
        normal = np.array(data_normal, copy=True)
        diagonal_indices = np.diag_indices(self.solution_size)
        for weight, regularization in regularization_terms:
            if regularization.is_diagonal:
                diagonal = np.asarray(regularization.diagonal(backend="numpy"))
                normal[diagonal_indices] += weight**2 * np.abs(diagonal) ** 2
                continue
            matrix = np.asarray(regularization.to_matrix(backend="numpy"))
            normal += weight**2 * (matrix.T.conj() @ matrix)
        return get_array_module().asarray(normal)

    def _custom_data_normal_matrix(self) -> np.ndarray:
        """Build and validate a supplied data-term normal matrix."""
        if self._data_normal_matrix_cache is None:
            matrix = np.asarray(self.data_normal_matrix_builder())
            expected_shape = (self.solution_size, self.solution_size)
            if matrix.shape != expected_shape:
                raise ValueError(
                    "data_normal_matrix_builder returned shape "
                    f"{matrix.shape}, expected {expected_shape}."
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError("The data normal matrix must contain only finite values.")
            self._data_normal_matrix_cache = matrix
        return self._data_normal_matrix_cache

    @cached_property
    def svd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the SVD of the dense system matrix."""
        return np.linalg.svd(self.system_matrix(backend="numpy"), full_matrices=False)

    def assemble_rhs_block(
        self, b: Any | list[Any], *, include_regularization: bool = True
    ) -> tuple[Any, tuple[int, ...], int]:
        """Assemble one or more right-hand side columns."""
        b_list = self._prepare_input_list(b, "b", count=self.num_data_terms)
        processed = [
            self._process_b_vector(b_val, self.data_shapes[i]) for i, b_val in enumerate(b_list)
        ]
        valid_b = [p for p in processed if p[0] is not None]
        if not valid_b:
            raise ValueError("At least one right-hand-side data term must be provided.")
        rhs_shape = valid_b[0][1]
        if not all(p[1] == rhs_shape for p in valid_b):
            raise ValueError("Inconsistent RHS column shapes in b terms.")

        num_rhs = math.prod(rhs_shape) if rhs_shape else 1
        dtype = self.A[0].dtype if self.A else np.float64
        active_regularization_terms = (
            self._active_regularization_terms() if include_regularization else ()
        )
        system_operator = self.system_operator if include_regularization else self.data_operator
        backend_operands = system_operator.backend_operands
        xp = get_array_module(*(p[0] for p in valid_b), *backend_operands)

        blocks = []
        for i, (b_col_block, _) in enumerate(processed):
            num_a_rows = self.A[i].shape[0]
            if b_col_block is None:
                blocks.append(xp.zeros((num_a_rows, num_rhs), dtype=dtype))
                continue
            w_item = self.sqrt_weights[i]
            if w_item is not None:
                b_col_block = w_item.matmat(b_col_block)
            blocks.append(xp.asarray(b_col_block).reshape(num_a_rows, num_rhs))

        for _, L_item in active_regularization_terms:
            blocks.append(xp.zeros((L_item.shape[0], num_rhs), dtype=dtype))

        d_block = xp.vstack(blocks) if blocks else xp.zeros((0, num_rhs), dtype=dtype)
        return d_block, rhs_shape, num_rhs

    @cached_property
    def system_operator(self) -> LinearMap:
        """Stack active regularization below the canonical data operator."""
        regularization_terms = self._active_regularization_terms()
        if not regularization_terms:
            return self.data_operator
        row_maps = [self.data_operator]
        row_maps.extend(weight * matrix for weight, matrix in regularization_terms)
        return vstack_linear_maps(row_maps, input_shape=self.solution_shape)

    def _active_regularization_terms(self) -> tuple[tuple[float, LinearMap], ...]:
        row_scales = self.regularization_row_scales
        return tuple(
            (row_scales[i], L_item)
            for i, L_item in enumerate(self.regularization_operators)
            if i < len(row_scales) and L_item is not None and row_scales[i] > 0.0
        )

    @staticmethod
    def _validate_regularization_strengths(weights: list) -> list[float]:
        """Return finite, non-negative scalar regularization weights."""
        validated = []
        for index, weight in enumerate(weights):
            try:
                array = np.asarray(weight)
                if array.ndim != 0:
                    raise ValueError
                value = float(array)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"regularization_strengths[{index}] must be a finite non-negative scalar."
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"regularization_strengths[{index}] must be a finite non-negative scalar."
                )
            validated.append(value)
        return validated

    @staticmethod
    def _prepare_input_list(
        item: Any | None,
        name: str,
        count: int | None = None,
        is_optional: bool = False,
        default_val: Any = None,
    ) -> list:
        if item is None:
            if is_optional:
                return []
            if count is None:
                raise ValueError(f"Input '{name}' cannot be None.")
            return [default_val] * count
        lst = item if isinstance(item, list) else [item]
        if count is not None and len(lst) != count:
            raise ValueError(f"Input '{name}' has {len(lst)} items, but expected {count}.")
        return lst

    def _normalize_data_shapes(
        self, data_shapes: Any, expected_count: int
    ) -> list[tuple[int, ...]]:
        if not isinstance(data_shapes, list):
            data_shapes = [data_shapes]
        if len(data_shapes) == 1 and expected_count > 1:
            data_shapes = data_shapes * expected_count
        if len(data_shapes) != expected_count:
            raise ValueError("Number of data_shapes does not match number of A operators.")
        return [(s,) if isinstance(s, int) else tuple(s) for s in data_shapes]

    def _process_b_vector(
        self, b_val: Any, data_shape: tuple[int, ...]
    ) -> tuple[Any | None, tuple[int, ...] | None]:
        if b_val is None:
            return None, None
        xp = get_array_module(b_val)
        b = xp.asarray(b_val)
        flat_data_size = math.prod(data_shape)
        num_data_dims = len(data_shape)

        if b.shape == data_shape:
            return b.reshape(flat_data_size, 1), ()

        if b.ndim > num_data_dims and tuple(b.shape[:num_data_dims]) == data_shape:
            rhs_shape = b.shape[num_data_dims:]
            return b.reshape(flat_data_size, math.prod(rhs_shape)), rhs_shape

        if b.ndim > num_data_dims and tuple(b.shape[-num_data_dims:]) == data_shape:
            rhs_shape = b.shape[:-num_data_dims]
            return b.reshape(math.prod(rhs_shape), flat_data_size).T, rhs_shape

        if b.ndim == 1 and b.size == flat_data_size:
            return b.reshape(flat_data_size, 1), ()

        if b.ndim == 0 and flat_data_size == 1:
            return b.reshape(1, 1), ()

        raise ValueError(f"Shape {b.shape} incompatible with data_shape {data_shape}.")
