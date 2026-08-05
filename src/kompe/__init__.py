"""Spherical representations and transforms."""

from importlib.metadata import PackageNotFoundError, version

from kompe.constants import EARTH_RADIUS_M, MU0
from kompe.core import (
    BasisView,
    ScalarBasis,
    SphericalBasis,
    SphericalRepresentation,
    SurfaceDifferentialBasis,
)
from kompe.cubed_sphere import (
    GlobalCSBasis,
    GlobalCSMesh,
    GlobalCSProjection,
    RegionalCSMesh,
    RegionalCSMeshSpec,
    RegionalCSOperators,
    RegionalCSProjection,
)
from kompe.grid import SphericalGrid
from kompe.mesh import StructuredSurfaceMesh
from kompe.secs import SECSBasis
from kompe.spherical_harmonics.sh_basis import SHBasis
from kompe.spherical_harmonics.solid_harmonics import SolidHarmonicOperators
from kompe.spherical_transform import SphericalTransform

try:
    __version__ = version("kompe")
except PackageNotFoundError:  # pragma: no cover - only an uninstalled source tree
    __version__ = "0+unknown"

__all__ = [
    "EARTH_RADIUS_M",
    "MU0",
    "BasisView",
    "GlobalCSBasis",
    "GlobalCSMesh",
    "GlobalCSProjection",
    "SphericalGrid",
    "RegionalCSMesh",
    "RegionalCSMeshSpec",
    "RegionalCSOperators",
    "RegionalCSProjection",
    "SECSBasis",
    "SHBasis",
    "ScalarBasis",
    "SolidHarmonicOperators",
    "SphericalBasis",
    "SphericalRepresentation",
    "SphericalTransform",
    "StructuredSurfaceMesh",
    "SurfaceDifferentialBasis",
    "__version__",
]
