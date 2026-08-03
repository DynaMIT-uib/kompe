"""Interfaces for cell-based meshes embedded in spherical surfaces."""

from abc import ABC, abstractmethod

import numpy as np


class StructuredSurfaceMesh(ABC):
    """Common geometry contract for structured meshes on a sphere.

    A structured mesh stores one value per cell and encodes neighbourhood
    topology through its logical array shape.  Coordinate angles are in
    degrees.  Cell areas use the square of the mesh radius, so a unit-sphere
    mesh has areas summing to ``4 * pi`` when it covers the full sphere.

    This interface deliberately does not apply to arbitrary evaluation grids:
    a collection of spherical sample points need not define cells or topology.
    """

    @property
    @abstractmethod
    def signature(self):
        """Stable identity for operator caches and persisted metadata."""

    @property
    @abstractmethod
    def mesh_shape(self) -> tuple[int, ...]:
        """Logical shape whose product is the number of mesh cells."""

    @property
    @abstractmethod
    def cell_center_theta(self):
        """Cell-centre colatitudes in degrees with shape ``mesh_shape``."""

    @property
    @abstractmethod
    def cell_center_phi(self):
        """Cell-centre longitudes in degrees with shape ``mesh_shape``."""

    @property
    @abstractmethod
    def cell_areas(self):
        """Positive cell areas with shape ``mesh_shape``."""

    @property
    def cell_count(self) -> int:
        """Number of cells in the mesh."""
        return int(np.prod(self.mesh_shape, dtype=int))

    def validate_mesh_metadata(self) -> None:
        """Validate the common structured-mesh geometry contract."""
        shape = tuple(int(length) for length in self.mesh_shape)
        if not shape or any(length <= 0 for length in shape):
            raise ValueError("mesh_shape must contain positive dimensions.")

        theta = np.asarray(self.cell_center_theta)
        phi = np.asarray(self.cell_center_phi)
        areas = np.asarray(self.cell_areas)
        for name, values in (
            ("cell_center_theta", theta),
            ("cell_center_phi", phi),
            ("cell_areas", areas),
        ):
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {values.shape}.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")

        if np.any((theta < 0.0) | (theta > 180.0)):
            raise ValueError("cell_center_theta must lie between 0 and 180 degrees.")
        if np.any(areas <= 0.0):
            raise ValueError("cell_areas must be strictly positive.")


__all__ = ["StructuredSurfaceMesh"]
