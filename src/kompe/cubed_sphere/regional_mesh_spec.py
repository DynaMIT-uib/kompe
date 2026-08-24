"""Versioned persistence schema for regional cubed-sphere meshes."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from kompe.cubed_sphere.regional_mesh import (
    RegionalCSMesh,
    _cell_size_pair,
    _mesh_shape,
    _uniform_edge_axis,
)
from kompe.cubed_sphere.regional_projection import RegionalCSProjection

REGIONAL_CS_MESH_SCHEMA = "kompe.regional_cs_mesh"
REGIONAL_CS_MESH_SCHEMA_VERSION = 1


def _coordinate_pair(name, values):
    """Return a pair of finite floating-point coordinates."""
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain two numbers") from error
    if len(result) != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite numbers")
    return result


@dataclass(frozen=True)
class RegionalCSMeshSpec:
    """Versioned, consumer-neutral specification for a regional CS mesh."""

    position: tuple[float, float]
    orientation: tuple[float, float]
    length: float
    width: float
    radius: float
    shape: tuple[int, int] | None = None
    cell_size: tuple[float, float] | None = None
    xi_edges: tuple[float, ...] | None = None
    eta_edges: tuple[float, ...] | None = None
    xi_shift: float = 0.0
    schema: str = REGIONAL_CS_MESH_SCHEMA
    version: int = REGIONAL_CS_MESH_SCHEMA_VERSION

    def __post_init__(self):
        """Validate and normalize the persisted mesh description."""
        if self.schema != REGIONAL_CS_MESH_SCHEMA:
            raise ValueError(f"Unsupported regional-mesh schema: {self.schema!r}")
        if self.version != REGIONAL_CS_MESH_SCHEMA_VERSION:
            raise ValueError(f"Unsupported regional-mesh schema version: {self.version!r}")

        position = _coordinate_pair("position", self.position)
        orientation = _coordinate_pair("orientation", self.orientation)
        if np.linalg.norm(orientation) == 0:
            raise ValueError("orientation must be non-zero")
        orientation_norm = np.linalg.norm(orientation)
        if not np.isclose(orientation_norm, 1.0, rtol=0.0, atol=1e-15):
            orientation = tuple(
                float(value) for value in np.asarray(orientation) / orientation_norm
            )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation", orientation)

        for name in ("length", "width", "radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, value)

        xi_shift = float(self.xi_shift)
        if not np.isfinite(xi_shift):
            raise ValueError("xi_shift must be finite")
        object.__setattr__(self, "xi_shift", xi_shift)

        shape = _mesh_shape(self.shape)
        cell_size = _cell_size_pair(self.cell_size)
        xi_edges = _uniform_edge_axis("xi_edges", self.xi_edges)
        eta_edges = _uniform_edge_axis("eta_edges", self.eta_edges)
        explicit_edges = xi_edges is not None or eta_edges is not None
        if explicit_edges and (xi_edges is None or eta_edges is None):
            raise ValueError("xi_edges and eta_edges must be provided together")
        mode_count = int(shape is not None) + int(cell_size is not None) + int(explicit_edges)
        if mode_count != 1:
            raise ValueError("Provide exactly one of shape, cell_size, or explicit edges")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(self, "xi_edges", xi_edges)
        object.__setattr__(self, "eta_edges", eta_edges)

    @classmethod
    def from_mesh(cls, mesh):
        """Create a specification from a canonical mesh."""
        if not isinstance(mesh, RegionalCSMesh):
            raise TypeError("mesh must be a RegionalCSMesh")
        explicit_edges = mesh.requested_shape is None and mesh.requested_cell_size is None
        return cls(
            position=tuple(mesh.projection.position),
            orientation=tuple(mesh.projection.orientation),
            length=mesh.length,
            width=mesh.width,
            radius=mesh.radius,
            shape=mesh.requested_shape,
            cell_size=mesh.requested_cell_size,
            xi_edges=mesh.xi_edges if explicit_edges else None,
            eta_edges=mesh.eta_edges if explicit_edges else None,
            xi_shift=0.0 if explicit_edges else mesh.xi_shift,
        )

    @classmethod
    def from_dict(cls, metadata):
        """Parse the versioned canonical mapping format."""
        if not isinstance(metadata, Mapping):
            raise TypeError("regional-mesh metadata must be a mapping")
        projection = metadata.get("projection")
        if not isinstance(projection, Mapping):
            raise TypeError("projection metadata must be a mapping")
        return cls(
            schema=metadata.get("schema"),
            version=metadata.get("version"),
            position=projection.get("position"),
            orientation=projection.get("orientation"),
            length=metadata.get("length"),
            width=metadata.get("width"),
            radius=metadata.get("radius"),
            shape=metadata.get("shape"),
            cell_size=metadata.get("cell_size"),
            xi_edges=metadata.get("xi_edges"),
            eta_edges=metadata.get("eta_edges"),
            xi_shift=metadata.get("xi_shift", 0.0),
        )

    def to_dict(self):
        """Return the stable JSON-compatible representation."""
        return {
            "schema": self.schema,
            "version": self.version,
            "projection": {
                "position": list(self.position),
                "orientation": list(self.orientation),
            },
            "length": self.length,
            "width": self.width,
            "radius": self.radius,
            "shape": None if self.shape is None else list(self.shape),
            "cell_size": None if self.cell_size is None else list(self.cell_size),
            "xi_edges": None if self.xi_edges is None else list(self.xi_edges),
            "eta_edges": None if self.eta_edges is None else list(self.eta_edges),
            "xi_shift": self.xi_shift,
        }

    def to_mesh(self):
        """Construct a regional cubed-sphere mesh from this specification."""
        projection = RegionalCSProjection(self.position, self.orientation)
        if self.xi_edges is not None:
            # Version-1 metadata allowed a shift together with stored edges.
            # Preserve that exact persisted meaning at the schema boundary;
            # RegionalCSMesh.from_edges() treats its edges as final geometry.
            return RegionalCSMesh(
                projection,
                self.length,
                self.width,
                radius=self.radius,
                xi_edges=self.xi_edges,
                eta_edges=self.eta_edges,
                xi_shift=self.xi_shift,
            )
        mesh_resolution = {"shape": self.shape}
        if self.cell_size is not None:
            eta_cell_size, xi_cell_size = self.cell_size
            mesh_resolution = {
                "xi_cell_size": xi_cell_size,
                "eta_cell_size": eta_cell_size,
            }
        return RegionalCSMesh(
            projection,
            self.length,
            self.width,
            radius=self.radius,
            xi_shift=self.xi_shift,
            **mesh_resolution,
        )


__all__ = [
    "REGIONAL_CS_MESH_SCHEMA",
    "REGIONAL_CS_MESH_SCHEMA_VERSION",
    "RegionalCSMeshSpec",
]
