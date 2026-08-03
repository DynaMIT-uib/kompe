"""Radial operations for spherical-harmonic coefficients."""

from kompe.core import SurfaceOperators, is_sh_basis
from kompe.math import as_linear_map
from kompe.math.backend import get_array_module


class SolidHarmonics:
    """Extend an SH basis with regular and irregular radial laws.

    The wrapped basis describes the angular coefficient space. This
    object describes how those coefficients change when the reference
    radius of the corresponding solid-harmonic expansion changes.

    The geomagnetic scalar-potential convention used here is

    ``V_regular(r; R) = R sum(q_nm(R) (r / R)^n Y_nm)``

    and

    ``V_irregular(r; R) = R sum(g_nm(R) (R / r)^(n + 1) Y_nm)``.

    The leading reference-radius factor means that changing the
    reference radius from ``start`` to ``end`` scales regular
    coefficients by ``(start / end)^(1 - n)`` and irregular coefficients
    by ``(start / end)^(n + 2)``.

    Poloidal coefficients ``m_nm`` are defined so that the radial
    field at the reference sphere is ``n(n + 1) m_nm Y_nm``. The
    corresponding regular and irregular scalar-potential coefficients
    are ``-(n + 1) m_nm`` and ``n m_nm``, respectively.

    Those conversion factors do not depend on reference radius, so the
    regular and irregular shifts also apply directly to corresponding
    poloidal coefficients on the corresponding radial branch.
    """

    def __init__(self, basis):
        """Initialize radial operations for an SH angular basis."""
        if not isinstance(basis, SurfaceOperators) or not is_sh_basis(basis):
            raise TypeError("SolidHarmonics requires an SH surface basis.")
        if not hasattr(basis, "n"):
            raise TypeError("SolidHarmonics basis must expose harmonic degrees as 'n'.")
        basis.validate_metadata()
        self.basis = basis

    @property
    def signature(self):
        """Return a stable signature for this radial extension."""
        return ("SOLID_HARMONICS", self.basis.signature)

    def regular_reference_shift(self, start, end):
        """Shift regular coefficients to a new reference radius."""
        return get_array_module().asarray((start / end) ** (1 - self.basis.n))

    def irregular_reference_shift(self, start, end):
        """Shift irregular coefficients to a new reference radius."""
        return get_array_module().asarray((start / end) ** (self.basis.n + 2))

    @property
    def poloidal_to_regular_potential_factor(self):
        """Map poloidal to regular potential coefficients."""
        return get_array_module().asarray(-(self.basis.n + 1))

    @property
    def poloidal_to_irregular_potential_factor(self):
        """Map poloidal to irregular potential coefficients."""
        return get_array_module().asarray(self.basis.n)

    @property
    def poloidal_to_boundary_potential_jump_factor(self):
        """Map poloidal coefficients to normalized potential jump."""
        return (
            self.poloidal_to_irregular_potential_factor - self.poloidal_to_regular_potential_factor
        )

    def poloidal_to_boundary_potential_jump(self, radius):
        """Map poloidal coefficients to the boundary potential jump."""
        return radius * self.poloidal_to_boundary_potential_jump_factor

    def get_regular_reference_shift_operator(self, start, end):
        """Return the regular reference-radius shift operator."""
        return as_linear_map(self.regular_reference_shift(start, end))

    def get_irregular_reference_shift_operator(self, start, end):
        """Return the irregular reference-radius shift operator."""
        return as_linear_map(self.irregular_reference_shift(start, end))

    def get_poloidal_to_regular_potential_operator(self):
        """Return the poloidal-to-regular-potential operator."""
        return as_linear_map(self.poloidal_to_regular_potential_factor)

    def get_poloidal_to_irregular_potential_operator(self):
        """Return the poloidal-to-irregular-potential operator."""
        return as_linear_map(self.poloidal_to_irregular_potential_factor)

    def get_poloidal_to_boundary_potential_jump_operator(self, radius):
        """Return the poloidal-to-boundary-potential-jump operator."""
        return as_linear_map(self.poloidal_to_boundary_potential_jump(radius))
