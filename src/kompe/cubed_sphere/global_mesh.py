"""Structured mesh geometry for the global cubed sphere."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from kompe.basis import _owned_readonly_array
from kompe.cubed_sphere.arrayutils import determinants_3x3
from kompe.cubed_sphere.cs_coordinates import (
    cube_to_cartesian,
    cube_to_spherical,
    face_coordinate,
    metric_tensor,
)
from kompe.cubed_sphere.global_projection import GlobalCSProjection
from kompe.grid import SphericalGrid
from kompe.math.backend import backend_context
from kompe.mesh import StructuredSurfaceMesh


@dataclass(frozen=True, init=False)
class GlobalCSMesh(StructuredSurfaceMesh):
    """Structured six-face cubed-sphere mesh for one grid resolution."""

    cells_per_face: int
    xi: np.ndarray
    eta: np.ndarray
    face: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    metric_tensor: np.ndarray
    sqrt_detg: np.ndarray
    _cell_areas: np.ndarray
    projection: GlobalCSProjection

    def __init__(self, cells_per_face):
        """Construct a unit-sphere mesh with a fixed face-edge resolution."""
        if isinstance(cells_per_face, bool) or not isinstance(cells_per_face, (int, np.integer)):
            raise TypeError("cells_per_face must be an integer")
        if cells_per_face <= 0:
            raise ValueError("Cubed sphere mesh dimension must be positive")

        cells_per_face = int(cells_per_face)
        # A mesh owns immutable host geometry. Device arrays belong to field
        # evaluation, not this one-time topology and cell-area construction.
        with backend_context("numpy"):
            k, i, j = self._gridpoints(cells_per_face)
            xi = face_coordinate(i[:, :-1, :-1] + 0.5, cells_per_face).reshape(-1)
            eta = face_coordinate(j[:, :-1, :-1] + 0.5, cells_per_face).reshape(-1)
            face = k[:, :-1, :-1].reshape(-1)
            _, theta, phi = cube_to_spherical(xi, eta, face, deg=True)
            cell_metric = metric_tensor(xi, eta)
            sqrt_detg = np.sqrt(determinants_3x3(cell_metric))
            cell_areas = self._compute_cell_areas(cells_per_face)

        for name, value in (
            ("cells_per_face", cells_per_face),
            ("xi", xi),
            ("eta", eta),
            ("face", face),
            ("theta", theta),
            ("phi", phi),
            ("metric_tensor", cell_metric),
            ("sqrt_detg", sqrt_detg),
            ("_cell_areas", cell_areas),
            ("projection", GlobalCSProjection()),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self):
        """Own immutable arrays used by basis and cache identity."""
        for name in (
            "xi",
            "eta",
            "face",
            "theta",
            "phi",
            "metric_tensor",
            "sqrt_detg",
            "_cell_areas",
        ):
            object.__setattr__(self, name, _owned_readonly_array(getattr(self, name)))
        self.validate_mesh_metadata()

    def __repr__(self):
        """Summarize the global mesh without printing its arrays."""
        return f"GlobalCSMesh(cells_per_face={self.cells_per_face}, size={self.size})"

    @property
    def signature(self):
        """Stable mesh identity for operators and caches."""
        return ("GLOBAL_CS_MESH", int(self.cells_per_face))

    @property
    def shape(self):
        """Logical ``(face, eta, xi)`` cell shape."""
        return (6, int(self.cells_per_face), int(self.cells_per_face))

    @cached_property
    def cell_centers(self):
        """Cell-centre coordinates and unit-sphere area weights."""
        return SphericalGrid(
            theta=self.theta,
            phi=self.phi,
            area_weights=self._cell_areas,
        )

    @property
    def cell_areas(self):
        """Cell areas on the unit sphere."""
        return self._cell_areas.reshape(self.shape)

    def coordinate(self, index):
        """Return xi/eta coordinate values for logical edge indices."""
        return face_coordinate(index, self.cells_per_face)

    def grid_line_indices(self, *, flat=False):
        """Return face, xi, and eta indices for all mesh grid lines."""
        face, xi, eta = self._gridpoints(self.cells_per_face)
        if flat:
            return face.reshape(-1), xi.reshape(-1), eta.reshape(-1)
        return face, xi, eta

    @staticmethod
    def _gridpoints(N):
        """Return face and grid-line indices for a mesh resolution."""
        return np.meshgrid(np.arange(6), np.arange(N + 1), np.arange(N + 1), indexing="ij")

    @staticmethod
    def spherical_triangle_area(a, b, c):
        """Return oriented unit-sphere triangle area magnitude."""
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            1.0
            + np.einsum("ij,ij->i", a, b)
            + np.einsum("ij,ij->i", b, c)
            + np.einsum("ij,ij->i", c, a)
        )
        return np.abs(2.0 * np.arctan2(numerator, denominator))

    @classmethod
    def _compute_cell_areas(cls, N):
        """Return exact spherical CS cell areas."""
        k, i, j = cls._gridpoints(N)
        block = k[:, :-1, :-1].reshape(-1)
        i0, i1 = i[:, :-1, :-1].reshape(-1), i[:, 1:, :-1].reshape(-1)
        j0, j1 = j[:, :-1, :-1].reshape(-1), j[:, :-1, 1:].reshape(-1)

        corners = [
            (face_coordinate(i0, N), face_coordinate(j0, N)),
            (face_coordinate(i1, N), face_coordinate(j0, N)),
            (face_coordinate(i1, N), face_coordinate(j1, N)),
            (face_coordinate(i0, N), face_coordinate(j1, N)),
        ]
        vectors = []
        for xi, eta in corners:
            x, y, z = cube_to_cartesian(xi, eta, np.ones_like(xi), block)
            vector = np.stack([x, y, z], axis=1)
            vectors.append(vector / np.linalg.norm(vector, axis=1).reshape((-1, 1)))

        return cls.spherical_triangle_area(
            vectors[0], vectors[1], vectors[2]
        ) + cls.spherical_triangle_area(vectors[0], vectors[2], vectors[3])


__all__ = ["GlobalCSMesh"]
