"""Implementation for einsum-backed ``LinearMap`` factories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kompe.math.backend import get_array_module, to_numpy
from kompe.math.linear_map import LinearMap, _normal_matrix_diag_from_matmat

_EINSUM_BATCH_LABELS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _dtype_of(value: Any):
    """Return dtype metadata without materializing backend arrays."""
    dtype = getattr(value, "dtype", None)
    return np.asarray(value).dtype if dtype is None else dtype


def _batched_einsum_string(einsum_string: str, operand_index: int) -> str | None:
    """Return an einsum string with one extra batch axis."""
    spec = einsum_string.replace(" ", "")
    if "..." in spec or "->" not in spec:
        return None
    lhs, rhs = spec.split("->", maxsplit=1)
    operands = lhs.split(",")
    if operand_index < 0:
        operand_index += len(operands)
    if operand_index < 0 or operand_index >= len(operands):
        return None

    used_labels = set(lhs.replace(",", "") + rhs)
    batch_label = next((label for label in _EINSUM_BATCH_LABELS if label not in used_labels), None)
    if batch_label is None:
        return None

    operands[operand_index] = operands[operand_index] + batch_label
    return ",".join(operands) + "->" + rhs + batch_label


def _parse_explicit_einsum(einsum_string: str) -> tuple[tuple[str, ...], str]:
    """Parse an explicit, non-ellipsis einsum specification."""
    spec = einsum_string.replace(" ", "")
    if "..." in spec or "->" not in spec:
        raise ValueError("Einsum composition requires explicit non-ellipsis strings.")
    lhs, rhs = spec.split("->", maxsplit=1)
    operands = tuple(lhs.split(",")) if lhs else ()
    return operands, rhs


def _dense_axis_labels(einsum_map: _EinsumMap) -> tuple[str, str]:
    """Return output and input labels from a dense einsum."""
    _, dense_output = _parse_explicit_einsum(einsum_map.einsum_string_dense)
    output_ndim = len(einsum_map.output_shape)
    input_ndim = len(einsum_map.input_shape)
    if len(dense_output) != output_ndim + input_ndim:
        raise ValueError("Dense einsum output does not match map shape metadata.")

    output_labels = dense_output[:output_ndim]
    input_labels = dense_output[output_ndim:]
    if len(set(output_labels)) != len(output_labels):
        raise ValueError("Dense output labels must be unique.")
    if len(set(input_labels)) != len(input_labels):
        raise ValueError("Dense input labels must be unique.")
    if set(output_labels).intersection(input_labels):
        raise ValueError("Dense input and output labels must be distinct.")
    return output_labels, input_labels


def _allocate_labels(count: int, used_labels: set[str]) -> str:
    """Allocate fresh single-character einsum labels."""
    if count == 0:
        return ""
    labels = []
    for label in _EINSUM_BATCH_LABELS:
        if label in used_labels:
            continue
        labels.append(label)
        used_labels.add(label)
        if len(labels) == count:
            return "".join(labels)
    raise ValueError("Not enough einsum labels available for fused composition.")


def _remap_subscript(subscript: str, label_map: dict[str, str], used_labels: set[str]) -> str:
    """Return ``subscript`` with collision-free labels."""
    remapped = []
    for label in subscript:
        if label not in label_map:
            label_map[label] = _allocate_labels(1, used_labels)
        remapped.append(label_map[label])
    return "".join(remapped)


def _remap_component_subscripts(
    component_subscripts: Sequence[str], label_map: dict[str, str], used_labels: set[str]
) -> tuple[str, ...]:
    """Remap all component tensor subscripts for one einsum map."""
    return tuple(
        _remap_subscript(subscript, label_map, used_labels) for subscript in component_subscripts
    )


def _remap_einsum_components(
    einsum_map: _EinsumMap, output_labels: str, input_labels: str, used_labels: set[str]
) -> tuple[str, ...]:
    """Remap map component labels to requested external labels."""
    component_subscripts, _ = _parse_explicit_einsum(einsum_map.einsum_string_dense)
    output_old, input_old = _dense_axis_labels(einsum_map)
    label_map = dict(zip(output_old, output_labels, strict=True))
    label_map.update(zip(input_old, input_labels, strict=True))
    return _remap_component_subscripts(component_subscripts, label_map, used_labels)


def compose_einsum_maps(left: _EinsumMap, right: _EinsumMap) -> _EinsumMap:
    """Return a fused einsum representation of ``left @ right``.

    The shaped domain of ``left`` must exactly match the shaped range of
    ``right``. Flat-only compatibility stays generic.
    """
    if left.input_shape != right.output_shape:
        raise ValueError("Einsum maps require matching shaped intermediate axes.")

    used_labels: set[str] = set()
    output_labels = _allocate_labels(len(left.output_shape), used_labels)
    input_labels = _allocate_labels(len(right.input_shape), used_labels)
    intermediate_labels = _allocate_labels(len(left.input_shape), used_labels)

    fused_components = _remap_einsum_components(
        left, output_labels, intermediate_labels, used_labels
    ) + _remap_einsum_components(right, intermediate_labels, input_labels, used_labels)

    dense_output = output_labels + input_labels
    einsum_string_dense = ",".join(fused_components) + "->" + dense_output
    einsum_string_matvec = ",".join(fused_components + (input_labels,)) + "->" + output_labels
    einsum_string_rmatvec = ",".join((output_labels,) + fused_components) + "->" + input_labels
    return _EinsumMap(
        component_tensors=left.component_tensors + right.component_tensors,
        einsum_string_dense=einsum_string_dense,
        einsum_string_matvec=einsum_string_matvec,
        einsum_string_rmatvec=einsum_string_rmatvec,
        output_shape=left.output_shape,
        input_shape=right.input_shape,
    )


def compose_diagonal_einsum_map(
    diagonal_values: Any, diagonal_shape: tuple[int, ...], einsum_map: _EinsumMap, *, side: str
) -> _EinsumMap:
    """Fuse a shaped diagonal scale with an einsum map."""
    diagonal_shape = tuple(diagonal_shape)
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    if side == "left" and diagonal_shape != einsum_map.output_shape:
        raise ValueError("Left diagonal shape must match einsum output shape.")
    if side == "right" and diagonal_shape != einsum_map.input_shape:
        raise ValueError("Right diagonal shape must match einsum input shape.")

    xp = get_array_module(diagonal_values)
    diagonal_tensor = xp.asarray(diagonal_values).reshape(diagonal_shape)

    used_labels: set[str] = set()
    output_labels = _allocate_labels(len(einsum_map.output_shape), used_labels)
    input_labels = _allocate_labels(len(einsum_map.input_shape), used_labels)
    remapped_components = _remap_einsum_components(
        einsum_map, output_labels, input_labels, used_labels
    )
    if side == "left":
        component_tensors = (diagonal_tensor,) + einsum_map.component_tensors
        component_subscripts = (output_labels,) + remapped_components
    else:
        component_tensors = einsum_map.component_tensors + (diagonal_tensor,)
        component_subscripts = remapped_components + (input_labels,)

    dense_output = output_labels + input_labels
    einsum_string_dense = ",".join(component_subscripts) + "->" + dense_output
    einsum_string_matvec = ",".join(component_subscripts + (input_labels,)) + "->" + output_labels
    einsum_string_rmatvec = ",".join((output_labels,) + component_subscripts) + "->" + input_labels
    return _EinsumMap(
        component_tensors=component_tensors,
        einsum_string_dense=einsum_string_dense,
        einsum_string_matvec=einsum_string_matvec,
        einsum_string_rmatvec=einsum_string_rmatvec,
        output_shape=einsum_map.output_shape,
        input_shape=einsum_map.input_shape,
    )


def dense_tensor_einsum_map(
    tensor: Any, *, output_shape: tuple[int, ...], input_shape: tuple[int, ...]
) -> _EinsumMap:
    """Return an einsum representation of an explicit dense tensor."""
    used_labels: set[str] = set()
    labels = _allocate_labels(len(output_shape) + len(input_shape), used_labels)
    output_labels = labels[: len(output_shape)]
    input_labels = labels[len(output_shape) :]
    return _EinsumMap(
        component_tensors=(tensor,),
        einsum_string_dense=f"{labels}->{labels}",
        einsum_string_matvec=f"{labels},{input_labels}->{output_labels}",
        einsum_string_rmatvec=f"{output_labels},{labels}->{input_labels}",
        output_shape=output_shape,
        input_shape=input_shape,
    )


def scale_einsum_map(einsum_map: _EinsumMap, scalar: Any) -> _EinsumMap:
    """Return an einsum map representing ``scalar * einsum_map``."""
    xp = get_array_module(scalar, *einsum_map.component_tensors)
    scalar_tensor = xp.asarray(scalar)
    component_subscripts, dense_output = _parse_explicit_einsum(einsum_map.einsum_string_dense)
    output_labels, input_labels = _dense_axis_labels(einsum_map)
    scaled_components = ("",) + component_subscripts

    return _EinsumMap(
        component_tensors=(scalar_tensor,) + einsum_map.component_tensors,
        einsum_string_dense=",".join(scaled_components) + "->" + dense_output,
        einsum_string_matvec=(
            ",".join(scaled_components + (input_labels,)) + "->" + output_labels
        ),
        einsum_string_rmatvec=(
            ",".join((output_labels,) + scaled_components) + "->" + input_labels
        ),
        output_shape=einsum_map.output_shape,
        input_shape=einsum_map.input_shape,
    )


def _derive_einsum_strings_from_matvec(
    einsum_string_matvec: str, num_component_tensors: int, input_operand_index: int
) -> tuple[str, str]:
    """Derive dense and adjoint einsum strings from a forward map."""
    spec = einsum_string_matvec.replace(" ", "")
    if "..." in spec or "->" not in spec:
        raise ValueError("Derived einsum maps require an explicit non-ellipsis output.")

    lhs, output_subscript = spec.split("->", maxsplit=1)
    operand_subscripts = lhs.split(",")
    if input_operand_index < 0:
        input_operand_index += len(operand_subscripts)
    if input_operand_index < 0 or input_operand_index >= len(operand_subscripts):
        raise ValueError("input_operand_index is outside the einsum operands.")
    if len(operand_subscripts) != num_component_tensors + 1:
        raise ValueError(
            "Forward einsum must contain all component tensors plus one input operand."
        )

    input_subscript = operand_subscripts[input_operand_index]
    component_subscripts = tuple(
        subscript
        for index, subscript in enumerate(operand_subscripts)
        if index != input_operand_index
    )
    if set(output_subscript).intersection(input_subscript):
        raise ValueError("Input and output subscripts must use distinct labels.")

    dense_subscript = ",".join(component_subscripts) + "->" + output_subscript + input_subscript
    rmatvec_subscript = (
        ",".join((output_subscript,) + component_subscripts) + "->" + input_subscript
    )
    return dense_subscript, rmatvec_subscript


@dataclass
class _EinsumMap:
    """Einsum implementation backing one ``LinearMap``."""

    component_tensors: tuple[Any, ...]
    einsum_string_dense: str
    einsum_string_matvec: str
    einsum_string_rmatvec: str
    output_shape: tuple[int, ...]
    input_shape: tuple[int, ...]
    _einsum_path_matvec: list | None = field(default=None, repr=False)
    _einsum_path_rmatvec: list | None = field(default=None, repr=False)
    _einsum_path_matmat: list | None = field(default=None, repr=False)
    _einsum_path_rmatmat: list | None = field(default=None, repr=False)
    _einsum_path_normal_diag: list | None = field(default=None, repr=False)
    _einsum_string_matmat: str | None = field(default=None, repr=False)
    _einsum_string_rmatmat: str | None = field(default=None, repr=False)
    _einsum_string_normal_diag: str | None = field(default=None, repr=False)
    _component_arrays_np: list[np.ndarray] | None = field(default=None, repr=False)

    @property
    def dtype(self):
        """Data type of the operator, given by its component tensors."""
        return np.result_type(*[_dtype_of(tensor) for tensor in self.component_tensors])

    def to_linear_map(self) -> LinearMap:
        """Return this einsum contraction as a ``LinearMap``."""
        flat_out = math.prod(self.output_shape)
        flat_in = math.prod(self.input_shape)
        return LinearMap(
            shape=(flat_out, flat_in),
            dtype=self.dtype,
            _matvec=self.matvec,
            _rmatvec=self.rmatvec,
            _matmat=self.matmat,
            _rmatmat=self.rmatmat,
            _dense_array_func=self.dense_array,
            _normal_matrix_diag=self.normal_matrix_diag,
            _backend_operands=self.component_tensors,
            _einsum_map=self,
            output_shape=self.output_shape,
            input_shape=self.input_shape,
        )

    def dense_array(self, xp: Any = None) -> Any:
        """Return dense matrix on the requested backend."""
        xp = get_array_module(*self.component_tensors) if xp is None else xp
        component_arrays = [xp.asarray(tensor) for tensor in self.component_tensors]
        dense_matrix = xp.einsum(self.einsum_string_dense, *component_arrays, optimize=True)
        return dense_matrix.reshape(math.prod(self.output_shape), math.prod(self.input_shape))

    def normal_matrix_diag(self) -> np.ndarray:
        """Compute ``diag(A* A)`` without building the dense matrix."""
        try:
            return self._normal_matrix_diag_einsum()
        except ValueError:
            return self._normal_matrix_diag_probe()

    def _normal_matrix_diag_probe(self) -> np.ndarray:
        """Compute ``diag(A* A)`` by applying identity blocks."""
        shape = (math.prod(self.output_shape), math.prod(self.input_shape))
        return _normal_matrix_diag_from_matmat(shape, self.dtype, self.matmat)

    def _normal_diag_string(self) -> str:
        """Return an einsum string for ``diag(A* A)``."""
        if self._einsum_string_normal_diag is None:
            used_labels: set[str] = set()
            output_labels = _allocate_labels(len(self.output_shape), used_labels)
            input_labels = _allocate_labels(len(self.input_shape), used_labels)
            conj_components = _remap_einsum_components(
                self, output_labels, input_labels, used_labels
            )
            components = _remap_einsum_components(self, output_labels, input_labels, used_labels)
            self._einsum_string_normal_diag = (
                ",".join(conj_components + components) + "->" + input_labels
            )
        return self._einsum_string_normal_diag

    def _normal_diag_path(self) -> list:
        """Return the cached optimized NumPy path for ``diag(A* A)``."""
        if self._einsum_path_normal_diag is None:
            component_arrays = self._numpy_component_arrays()
            conj_arrays = [arr.conj() for arr in component_arrays]
            self._einsum_path_normal_diag = np.einsum_path(
                self._normal_diag_string(), *conj_arrays, *component_arrays, optimize="greedy"
            )[0]
        return self._einsum_path_normal_diag

    def _normal_matrix_diag_einsum(self) -> np.ndarray:
        """Compute ``diag(A* A)`` as a direct tensor contraction."""
        component_arrays = self._numpy_component_arrays()
        conj_arrays = [arr.conj() for arr in component_arrays]
        diag = np.einsum(
            self._normal_diag_string(),
            *conj_arrays,
            *component_arrays,
            optimize=self._normal_diag_path(),
        )
        return np.asarray(diag).reshape(-1).real

    def _numpy_component_arrays(self) -> list[np.ndarray]:
        """Return cached NumPy component arrays."""
        if self._component_arrays_np is None:
            self._component_arrays_np = [to_numpy(t) for t in self.component_tensors]
        return self._component_arrays_np

    def _matvec_path(self) -> list:
        """Return the cached optimized NumPy path for matvec."""
        if self._einsum_path_matvec is None:
            dummy_input = np.empty(self.input_shape, dtype=self.dtype)
            self._einsum_path_matvec = np.einsum_path(
                self.einsum_string_matvec,
                *self._numpy_component_arrays(),
                dummy_input,
                optimize="greedy",
            )[0]
        return self._einsum_path_matvec

    def _rmatvec_path(self) -> list:
        """Return the cached optimized NumPy path for rmatvec."""
        if self._einsum_path_rmatvec is None:
            dummy_grad_output = np.empty(self.output_shape, dtype=self.dtype)
            self._einsum_path_rmatvec = np.einsum_path(
                self.einsum_string_rmatvec,
                dummy_grad_output,
                *self._numpy_component_arrays(),
                optimize="greedy",
            )[0]
        return self._einsum_path_rmatvec

    def _matmat_string(self) -> str | None:
        """Return a batched matvec einsum string if possible."""
        if self._einsum_string_matmat is None:
            self._einsum_string_matmat = _batched_einsum_string(self.einsum_string_matvec, -1)
        return self._einsum_string_matmat

    def _rmatmat_string(self) -> str | None:
        """Return a batched adjoint einsum string if possible."""
        if self._einsum_string_rmatmat is None:
            self._einsum_string_rmatmat = _batched_einsum_string(self.einsum_string_rmatvec, 0)
        return self._einsum_string_rmatmat

    def _matmat_path(self) -> list | None:
        """Return the cached optimized NumPy path for matmat."""
        einsum_string = self._matmat_string()
        if einsum_string is None:
            return None
        if self._einsum_path_matmat is None:
            dummy_input = np.empty(self.input_shape + (1,), dtype=self.dtype)
            self._einsum_path_matmat = np.einsum_path(
                einsum_string, *self._numpy_component_arrays(), dummy_input, optimize="greedy"
            )[0]
        return self._einsum_path_matmat

    def _rmatmat_path(self) -> list | None:
        """Return the cached optimized NumPy path for rmatmat."""
        einsum_string = self._rmatmat_string()
        if einsum_string is None:
            return None
        if self._einsum_path_rmatmat is None:
            dummy_grad_output = np.empty(self.output_shape + (1,), dtype=self.dtype)
            self._einsum_path_rmatmat = np.einsum_path(
                einsum_string,
                dummy_grad_output,
                *self._numpy_component_arrays(),
                optimize="greedy",
            )[0]
        return self._einsum_path_rmatmat

    def _matvec_numpy(self, x_flat: Any) -> np.ndarray:
        """Apply using cached NumPy contraction paths."""
        x_tensor = np.asarray(x_flat).reshape(self.input_shape)
        res = np.einsum(
            self.einsum_string_matvec,
            *self._numpy_component_arrays(),
            x_tensor,
            optimize=self._matvec_path(),
        )
        return res.reshape(-1)

    def _rmatvec_numpy(self, y_flat: Any) -> np.ndarray:
        """Apply the adjoint using cached NumPy contraction paths."""
        grad_tensor = np.asarray(y_flat).reshape(self.output_shape)
        conj_tensors = [arr.conj() for arr in self._numpy_component_arrays()]
        grad_x = np.einsum(
            self.einsum_string_rmatvec, grad_tensor, *conj_tensors, optimize=self._rmatvec_path()
        )
        return grad_x.reshape(-1)

    def _matmat_numpy(self, x_block: Any) -> np.ndarray | None:
        """Apply multiple vectors with cached NumPy paths."""
        einsum_string = self._matmat_string()
        einsum_path = self._matmat_path()
        if einsum_string is None or einsum_path is None:
            return None
        block = np.asarray(x_block)
        x_tensor = block.reshape(self.input_shape + (block.shape[1],))
        res = np.einsum(
            einsum_string, *self._numpy_component_arrays(), x_tensor, optimize=einsum_path
        )
        return res.reshape(math.prod(self.output_shape), block.shape[1])

    def _rmatmat_numpy(self, y_block: Any) -> np.ndarray | None:
        """Apply adjoints using cached NumPy paths."""
        einsum_string = self._rmatmat_string()
        einsum_path = self._rmatmat_path()
        if einsum_string is None or einsum_path is None:
            return None
        block = np.asarray(y_block)
        grad_tensor = block.reshape(self.output_shape + (block.shape[1],))
        conj_tensors = [arr.conj() for arr in self._numpy_component_arrays()]
        grad_x = np.einsum(einsum_string, grad_tensor, *conj_tensors, optimize=einsum_path)
        return grad_x.reshape(math.prod(self.input_shape), block.shape[1])

    def matvec(self, x_flat: Any) -> Any:
        """Apply the tensor contraction to one flattened vector."""
        xp = get_array_module(x_flat, *self.component_tensors)
        if xp is np:
            return self._matvec_numpy(x_flat)
        component_arrays = [xp.asarray(t) for t in self.component_tensors]
        x_tensor = xp.asarray(x_flat).reshape(self.input_shape)
        res = xp.einsum(self.einsum_string_matvec, *component_arrays, x_tensor, optimize=True)
        return xp.reshape(res, (-1,))

    def rmatvec(self, y_flat: Any) -> Any:
        """Apply the adjoint contraction to one flat vector."""
        xp = get_array_module(y_flat, *self.component_tensors)
        if xp is np:
            return self._rmatvec_numpy(y_flat)
        grad_tensor = xp.asarray(y_flat).reshape(self.output_shape)
        conj_tensors = [xp.conjugate(xp.asarray(t)) for t in self.component_tensors]
        grad_x = xp.einsum(self.einsum_string_rmatvec, grad_tensor, *conj_tensors, optimize=True)
        return xp.reshape(grad_x, (-1,))

    def matmat(self, x_block: Any) -> Any:
        """Apply the tensor contraction to multiple vectors."""
        xp = get_array_module(x_block, *self.component_tensors)
        x_arr = xp.asarray(x_block)
        if x_arr.ndim == 1:
            return self.matvec(x_arr)
        if xp is np:
            batched = self._matmat_numpy(x_arr)
            if batched is not None:
                return batched
        einsum_string = self._matmat_string()
        if einsum_string is not None:
            component_arrays = [xp.asarray(t) for t in self.component_tensors]
            x_tensor = x_arr.reshape(self.input_shape + (x_arr.shape[1],))
            res = xp.einsum(einsum_string, *component_arrays, x_tensor, optimize=True)
            return xp.reshape(res, (-1, x_arr.shape[1]))
        outputs = [self.matvec(x_arr[:, i]) for i in range(x_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        """Apply the adjoint tensor contraction to multiple vectors."""
        xp = get_array_module(y_block, *self.component_tensors)
        y_arr = xp.asarray(y_block)
        if y_arr.ndim == 1:
            return self.rmatvec(y_arr)
        if xp is np:
            batched = self._rmatmat_numpy(y_arr)
            if batched is not None:
                return batched
        einsum_string = self._rmatmat_string()
        if einsum_string is not None:
            grad_tensor = y_arr.reshape(self.output_shape + (y_arr.shape[1],))
            conj_tensors = [xp.conjugate(xp.asarray(t)) for t in self.component_tensors]
            grad_x = xp.einsum(einsum_string, grad_tensor, *conj_tensors, optimize=True)
            return xp.reshape(grad_x, (-1, y_arr.shape[1]))
        outputs = [self.rmatvec(y_arr[:, i]) for i in range(y_arr.shape[1])]
        return xp.stack(outputs, axis=1)


def einsum_linear_map(
    *,
    component_tensors: Sequence[Any],
    einsum_string_dense: str,
    einsum_string_matvec: str,
    einsum_string_rmatvec: str,
    output_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
) -> LinearMap:
    """Return a ``LinearMap`` backed by cached einsum contractions."""
    return _EinsumMap(
        component_tensors=tuple(component_tensors),
        einsum_string_dense=einsum_string_dense,
        einsum_string_matvec=einsum_string_matvec,
        einsum_string_rmatvec=einsum_string_rmatvec,
        output_shape=tuple(output_shape),
        input_shape=tuple(input_shape),
    ).to_linear_map()


def einsum_linear_map_from_matvec(
    *,
    component_tensors: Sequence[Any],
    einsum_string_matvec: str,
    output_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    input_operand_index: int = -1,
) -> LinearMap:
    """Return an einsum-backed map from one forward contraction."""
    component_tensors = tuple(component_tensors)
    einsum_string_dense, einsum_string_rmatvec = _derive_einsum_strings_from_matvec(
        einsum_string_matvec, len(component_tensors), input_operand_index
    )
    return einsum_linear_map(
        component_tensors=component_tensors,
        einsum_string_dense=einsum_string_dense,
        einsum_string_matvec=einsum_string_matvec,
        einsum_string_rmatvec=einsum_string_rmatvec,
        output_shape=output_shape,
        input_shape=input_shape,
    )
