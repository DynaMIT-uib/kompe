"""Coefficient layouts and gauge policies for ordinary arrays."""

from dataclasses import dataclass

from kompe.basis import ScalarBasis, SurfaceDifferentialBasis
from kompe.math import get_array_module


@dataclass(frozen=True)
class CoefficientSpace:
    """Describe the coefficient space for one field.

    ``CoefficientSpace`` deliberately carries no values and does not evaluate
    fields on grids. It describes a scalar or Helmholtz coefficient
    layout and whether values should satisfy a mean-free gauge.
    ``mean_free`` defaults to whether the basis already represents only
    zero-mean fields; setting it to True requests constant subtraction.
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
                self.basis.mean_free if isinstance(self.basis, SurfaceDifferentialBasis) else False
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
        """Normalize coefficients with scientific axes first and batch axes last."""
        array = self.validate_coefficients(coeffs, name=name)
        if not self.mean_free:
            return array

        if self.representation == "scalar":
            return self.basis.project_scalar_mean_free(array)
        return self.basis.project_helmholtz_mean_free(array)

    def validate_coefficients(self, coeffs, *, name="coefficients"):
        """Retain trailing batch axes and normalize leading field axes.

        Fields start with ``shape`` or one flattened axis of length ``size``.
        A single field may also be supplied in a grid-shaped array.
        """
        xp = get_array_module(coeffs)
        array = xp.asarray(coeffs)
        shape = self.shape
        if array.shape[: len(shape)] == shape:
            return array
        if array.ndim and array.shape[0] == self.size:
            return array.reshape(shape + array.shape[1:])
        if array.size != self.size:
            raise ValueError(
                f"{name} has {array.size} coefficients, expected "
                f"{self.size} for {self.representation} "
                f"{self.kind} field space."
            )
        return array.reshape(self.shape)


__all__ = ["CoefficientSpace"]
