"""Coefficient-space descriptors and realized field values."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from kompe.basis import ScalarBasis, SurfaceDifferentialBasis
from kompe.math import get_array_module


@dataclass(frozen=True)
class CoefficientSpace:
    """Describe the coefficient space for one field.

    ``CoefficientSpace`` deliberately carries no values and does not evaluate
    fields on grids. It describes a scalar or Helmholtz coefficient
    layout and whether values should satisfy a mean-free gauge.
    Helmholtz components store curl-free and divergence-free potentials,
    not the physical theta/phi field components.
    """

    basis: ScalarBasis
    representation: str = "scalar"
    mean_free: bool | None = None

    def __post_init__(self):
        """Validate field-space metadata."""
        if self.representation not in {"scalar", "helmholtz"}:
            raise ValueError("representation must be either 'scalar' or 'helmholtz'.")
        if not isinstance(self.basis, ScalarBasis):
            raise TypeError("CoefficientSpace basis must be a Kompe ScalarBasis.")
        if self.mean_free is None:
            mean_free = (
                self.basis.omits_constant_mode()
                if isinstance(self.basis, SurfaceDifferentialBasis)
                else False
            )
        else:
            mean_free = bool(self.mean_free)
        object.__setattr__(self, "mean_free", mean_free)
        if (self.representation == "helmholtz" or self.mean_free) and not isinstance(
            self.basis, SurfaceDifferentialBasis
        ):
            raise TypeError(
                f"A {self.representation} CoefficientSpace with mean_free={self.mean_free} "
                "requires a SurfaceDifferentialBasis."
            )

    @property
    def kind(self):
        """Return the underlying basis-family identifier."""
        return self.basis.kind

    @property
    def index_names(self):
        """Return coefficient index names."""
        return self.basis.index_names

    @property
    def index_arrays(self):
        """Return per-coefficient index arrays."""
        return self.basis.index_arrays

    @property
    def coefficient_count(self):
        """Return scalar coefficient count."""
        return int(self.basis.coefficient_count)

    @property
    def component_count(self):
        """Return coefficient component count for this field type."""
        return 2 if self.representation == "helmholtz" else 1

    @property
    def size(self):
        """Return flattened coefficient count for one variable."""
        return self.component_count * self.coefficient_count

    @property
    def shape(self):
        """Return canonical coefficient array shape for one variable."""
        if self.representation == "scalar":
            return (self.coefficient_count,)
        return (self.component_count, self.coefficient_count)

    @property
    def signature(self):
        """Return coefficient-layout and storage-policy identity."""
        return (self.basis.coefficient_space_signature, self.representation, bool(self.mean_free))

    def project_mean_free(self, coeffs, *, name="coefficients"):
        """Apply this space's mean-free coefficient policy."""
        array = self.validate_coefficients(coeffs, name=name)
        if not self.mean_free:
            return array

        if self.representation == "scalar":
            return self.basis.project_scalar_mean_free(array)
        return self.basis.project_helmholtz_mean_free(array)

    def validate_coefficients(self, coeffs, *, name="coefficients"):
        """Return coefficients as an array after length validation."""
        xp = get_array_module(coeffs)
        array = xp.asarray(coeffs)
        if array.size != self.size:
            raise ValueError(
                f"{name} has {array.size} coefficients, expected "
                f"{self.size} for {self.representation} "
                f"{self.kind} field space."
            )
        return array.reshape(self.shape)


class FieldCoefficients:
    """Realized coefficient values in a :class:`CoefficientSpace`.

    The container owns its values, validates their shape, and applies
    the field space's gauge policy. Sampling and projection remain the
    coefficient basis's responsibility.
    """

    def __init__(self, field_space: CoefficientSpace, coeffs: Any, *, name: str | None = None):
        """Initialize owned field coefficients."""
        if not isinstance(field_space, CoefficientSpace):
            raise TypeError("FieldCoefficients requires a CoefficientSpace.")
        self.field_space = field_space
        field_name = name or f"{self.__class__.__name__}.array"
        # JAX on CPU may share a NumPy buffer instead of copying it.
        if isinstance(coeffs, np.ndarray) and get_array_module(coeffs) is not np:
            coeffs = np.array(coeffs, copy=True)
        array = self.field_space.project_mean_free(coeffs, name=field_name)
        if isinstance(array, np.ndarray):
            array = np.array(array, copy=True)
            array.setflags(write=False)
        self._array = array

    @property
    def array(self):
        """Return coefficients in canonical shaped form."""
        return self._array

    def __repr__(self):
        """Summarize coefficients without printing the full array."""
        return (
            f"FieldCoefficients(field_space={self.field_space!r}, "
            f"shape={self.array.shape}, dtype={self.array.dtype})"
        )

    def to_vector(self):
        """Return coefficients as a flat operator-compatible vector."""
        return self.array.reshape(-1)

    def __array__(self, dtype=None, copy=None):
        """Return coefficients for NumPy coercion."""
        if copy is None:
            return np.asarray(self.array, dtype=dtype)
        return np.array(self.array, dtype=dtype, copy=copy)


__all__ = ["CoefficientSpace", "FieldCoefficients"]
