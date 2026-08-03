"""Spherical and Cartesian elementary current-system representations."""

from kompe.constants import EARTH_RADIUS_M
from kompe.secs.basis import (
    DEFAULT_IONOSPHERE_RADIUS_M,
    SECSBasis,
)
from kompe.secs.cartesian import (
    azimuth_unit_vectors,
    cartesian_current_matrices,
    cartesian_distance,
    cartesian_magnetic_field_matrices,
    horizontal_azimuth,
    horizontal_distance,
    radial_unit_vectors,
    relative_coordinates,
)
from kompe.secs.kernels import (
    angular_distance,
    current_wedge_magnetic_field_matrix,
    inclined_field_magnetic_field_matrices,
    magnetic_field_matrices,
    surface_current_matrices,
)

__all__ = [
    "DEFAULT_IONOSPHERE_RADIUS_M",
    "EARTH_RADIUS_M",
    "SECSBasis",
    "angular_distance",
    "azimuth_unit_vectors",
    "cartesian_current_matrices",
    "cartesian_distance",
    "cartesian_magnetic_field_matrices",
    "current_wedge_magnetic_field_matrix",
    "horizontal_azimuth",
    "horizontal_distance",
    "inclined_field_magnetic_field_matrices",
    "magnetic_field_matrices",
    "radial_unit_vectors",
    "relative_coordinates",
    "surface_current_matrices",
]
