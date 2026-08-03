"""Spherical representations and transforms."""

from importlib.metadata import PackageNotFoundError, version

from kompe.constants import EARTH_RADIUS_M, MU0
from kompe.core import (
    BasisView,
    ScalarSynthesis,
    SphericalBasis,
    SphericalRepresentation,
    SurfaceOperators,
    basis_kind,
    is_basis_kind,
    is_cs_basis,
    is_secs_basis,
    is_sh_basis,
)
from kompe.cubed_sphere import (
    REGIONAL_CS_GRID_SCHEMA,
    REGIONAL_CS_GRID_SCHEMA_VERSION,
    GlobalCSBasis,
    GlobalCSMesh,
    RegionalCSGrid,
    RegionalCSGridSpec,
    RegionalCSOperators,
    RegionalCSProjection,
)
from kompe.grid import Grid
from kompe.mesh import StructuredSurfaceMesh
from kompe.secs import SECSBasis
from kompe.spherical_harmonics.sh_basis import SHBasis
from kompe.spherical_harmonics.solid_harmonics import SolidHarmonics
from kompe.spherical_transform import SphericalTransform

try:
    __version__ = version("kompe")
except PackageNotFoundError:  # pragma: no cover - only an uninstalled source tree
    __version__ = "0+unknown"

__all__ = [
    "EARTH_RADIUS_M",
    "MU0",
    "REGIONAL_CS_GRID_SCHEMA",
    "REGIONAL_CS_GRID_SCHEMA_VERSION",
    "BasisView",
    "GlobalCSBasis",
    "GlobalCSMesh",
    "Grid",
    "RegionalCSGrid",
    "RegionalCSGridSpec",
    "RegionalCSOperators",
    "RegionalCSProjection",
    "SECSBasis",
    "SHBasis",
    "ScalarSynthesis",
    "SolidHarmonics",
    "SphericalBasis",
    "SphericalRepresentation",
    "SphericalTransform",
    "StructuredSurfaceMesh",
    "SurfaceOperators",
    "__version__",
    "basis_kind",
    "is_basis_kind",
    "is_cs_basis",
    "is_secs_basis",
    "is_sh_basis",
]
