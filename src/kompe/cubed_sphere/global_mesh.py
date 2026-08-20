"""Native-grid and remapping helpers for CS surface bases."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

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
from kompe.math import as_linear_map, identity_linear_map
from kompe.math.backend import jax_enabled, to_numpy
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
        k, i, j = self._gridpoints(cells_per_face)
        xi = face_coordinate(i[:, :-1, :-1] + 0.5, cells_per_face).reshape(-1)
        eta = face_coordinate(j[:, :-1, :-1] + 0.5, cells_per_face).reshape(-1)
        face = k[:, :-1, :-1].reshape(-1)
        _, theta, phi = cube_to_spherical(xi, eta, face, deg=True)
        cell_metric = metric_tensor(xi, eta)

        for name, value in (
            ("cells_per_face", cells_per_face),
            ("xi", xi),
            ("eta", eta),
            ("face", face),
            ("theta", theta),
            ("phi", phi),
            ("metric_tensor", cell_metric),
            ("sqrt_detg", np.sqrt(determinants_3x3(cell_metric))),
            ("_cell_areas", self._compute_cell_areas(cells_per_face)),
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


class _GlobalCSRemapper:
    """Build and cache remaps between CS-compatible grids."""

    _shared_remap_matrix_cache: ClassVar[OrderedDict] = OrderedDict()
    _shared_remap_matrix_cache_size: ClassVar[int] = 8
    _operator_cache_size: ClassVar[int] = 16

    def __init__(self, basis):
        self.basis = basis
        self.operator_cache = OrderedDict()

    @classmethod
    def clear_shared_cache(cls):
        """Clear process-wide CS remapping matrices."""
        cls._shared_remap_matrix_cache.clear()

    @classmethod
    def shared_cache_info(cls):
        """Return process-wide remapping cache occupancy and limit."""
        return {
            "size": len(cls._shared_remap_matrix_cache),
            "max_size": cls._shared_remap_matrix_cache_size,
        }

    def clear_cache(self):
        """Clear operators owned by this remapper instance."""
        self.operator_cache.clear()

    def cache_info(self):
        """Return instance operator-cache occupancy and limit."""
        return {"size": len(self.operator_cache), "max_size": self._operator_cache_size}

    def _store_operator(self, key, operator):
        """Store one operator while enforcing the per-basis cache limit."""
        self.operator_cache[key] = operator
        self.operator_cache.move_to_end(key)
        while len(self.operator_cache) > self._operator_cache_size:
            self.operator_cache.popitem(last=False)

    @staticmethod
    def grid_theta_phi(grid):
        """Return flattened theta/phi coordinates."""
        return (
            np.asarray(to_numpy(grid.theta), dtype=float).reshape(-1),
            np.asarray(to_numpy(grid.phi), dtype=float).reshape(-1),
        )

    @staticmethod
    def grid_signature(grid):
        """Return a cache key for a grid."""
        signature = getattr(grid, "signature", None)
        if signature is None:
            raise TypeError("CS grid remapping requires SphericalGrid objects with signatures.")
        return signature

    def _cached_remap_matrix(self, key, build):
        """Return a bounded shared remap matrix cache entry."""
        cache = self._shared_remap_matrix_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        matrix = build()
        cache[key] = matrix
        if len(cache) > self._shared_remap_matrix_cache_size:
            cache.popitem(last=False)
        return matrix

    def remap_matrix_key(self, kind, source_grid, target_grid):
        """Return a shared remap-matrix cache key."""
        basis_type = type(self.basis)
        return (
            basis_type.__module__,
            basis_type.__qualname__,
            kind,
            self.grid_signature(source_grid),
            self.grid_signature(target_grid),
        )

    @staticmethod
    def linear_interpolation_weights(source_points, target_points):
        """Return Delaunay vertices and barycentric weights."""
        if source_points.shape[0] < 3:
            raise ValueError("At least three source points are required.")
        triangulation = Delaunay(source_points)
        simplex = triangulation.find_simplex(target_points)
        if np.any(simplex < 0):
            raise ValueError("Target points lie outside the source interpolation hull.")

        transform = triangulation.transform[simplex]
        delta = target_points - transform[:, 2]
        first_weights = np.einsum("nij,nj->ni", transform[:, :2], delta)
        weights = np.column_stack([first_weights, 1.0 - np.sum(first_weights, axis=1)])
        return triangulation.simplices[simplex], weights

    def block_interpolation_weights(self, theta, phi, theta_target, phi_target):
        """Return per-block interpolation weights."""
        basis = self.basis
        xi_target, eta_target, block_target = basis.mesh.projection.geographic_to_cube(
            phi_target, 90 - theta_target
        )
        xi_target = xi_target.reshape(-1)
        eta_target = eta_target.reshape(-1)
        block_target = block_target.reshape(-1)

        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
        blocks = []

        for block_index in range(6):
            target_index = np.flatnonzero(block_target == block_index)
            if target_index.size == 0:
                continue

            _, th0, ph0 = basis.mesh.projection.cube_to_spherical(0, 0, block_index, degrees=False)
            r0 = np.array(
                [np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)]
            ).reshape((-1, 1))
            source_mask = np.sum(r0 * r, axis=0) > 0
            source_index = np.flatnonzero(source_mask)

            xi_source, eta_source, _ = basis.mesh.projection.geographic_to_cube(
                phi, 90 - theta, face=block_index
            )
            source_points = np.column_stack([xi_source[source_mask], eta_source[source_mask]])
            target_points = np.column_stack([xi_target[target_index], eta_target[target_index]])
            vertices, weights = self.linear_interpolation_weights(source_points, target_points)
            blocks.append((block_index, target_index, source_index[vertices], weights))

        return blocks

    def build_scalar_grid_remap_matrix(self, source_grid, target_grid):
        """Build a sparse scalar grid remap."""
        theta, phi = self.grid_theta_phi(source_grid)
        theta_target, phi_target = self.grid_theta_phi(target_grid)
        blocks = self.block_interpolation_weights(theta, phi, theta_target, phi_target)

        rows = []
        cols = []
        data = []
        for _, target_index, source_vertices, weights in blocks:
            rows.append(np.repeat(target_index, 3))
            cols.append(source_vertices.reshape(-1))
            data.append(weights.reshape(-1))

        if rows:
            row = np.concatenate(rows)
            col = np.concatenate(cols)
            values = np.concatenate(data)
        else:
            row = col = np.array([], dtype=int)
            values = np.array([], dtype=float)
        return sp.coo_matrix((values, (row, col)), shape=(theta_target.size, theta.size)).tocsr()

    def build_tangential_grid_remap_matrix(self, source_grid, target_grid):
        """Build a sparse tangential grid remap."""
        basis = self.basis
        theta, phi = self.grid_theta_phi(source_grid)
        theta_target, phi_target = self.grid_theta_phi(target_grid)
        blocks = self.block_interpolation_weights(theta, phi, theta_target, phi_target)

        xi_source, eta_source, block_source = basis.mesh.projection.geographic_to_cube(
            phi, 90 - theta
        )
        source_transform = basis.mesh.projection.enu_to_cube_vector_matrix(
            xi_source, eta_source, radius=1, face=block_source
        )

        xi_target, eta_target, block_target = basis.mesh.projection.geographic_to_cube(
            phi_target, 90 - theta_target
        )
        target_transform = basis.mesh.projection.cube_to_enu_vector_matrix(
            xi_target, eta_target, radius=1, face=block_target
        )

        n_source = theta.size
        n_target = theta_target.size
        out_components = np.arange(2)
        rows = []
        cols = []
        data = []

        for block_index, target_index, source_vertices, weights in blocks:
            qij = basis.mesh.projection.face_to_face_vector_matrix(
                xi_source, eta_source, block_source, block_index
            )
            source_to_block = np.einsum("nij,njk->nik", qij, source_transform)
            source_coeff = source_to_block[source_vertices]
            source_coeff = np.stack([-source_coeff[..., 1], source_coeff[..., 0]], axis=-1)
            target_coeff = target_transform[target_index]
            target_coeff = np.stack([-target_coeff[:, 1, :], target_coeff[:, 0, :]], axis=1)

            coefficients = weights[:, :, None, None] * np.einsum(
                "tob,tvbi->tvoi", target_coeff, source_coeff
            )
            row = target_index[:, None, None, None] + (
                out_components[None, None, :, None] * n_target
            )
            col = source_vertices[:, :, None, None] + (
                out_components[None, None, None, :] * n_source
            )
            rows.append(np.broadcast_to(row, coefficients.shape).reshape(-1))
            cols.append(np.broadcast_to(col, coefficients.shape).reshape(-1))
            data.append(coefficients.reshape(-1))

        if rows:
            row = np.concatenate(rows)
            col = np.concatenate(cols)
            values = np.concatenate(data)
        else:
            row = col = np.array([], dtype=int)
            values = np.array([], dtype=float)
        return sp.coo_matrix((values, (row, col)), shape=(2 * n_target, 2 * n_source)).tocsr()

    def scalar_grid_remap_operator(self, source_grid, target_grid):
        """Return a cached scalar grid-remap operator."""
        if source_grid.same_as(target_grid):
            return identity_linear_map((source_grid.size,))
        matrix_key = self.remap_matrix_key("scalar_grid_remap_matrix", source_grid, target_grid)
        key = ("scalar_grid_remap", matrix_key, bool(jax_enabled()))
        if key not in self.operator_cache:
            matrix = self._cached_remap_matrix(
                matrix_key, lambda: self.build_scalar_grid_remap_matrix(source_grid, target_grid)
            )
            self._store_operator(
                key,
                as_linear_map(
                    matrix, input_shape=(source_grid.size,), output_shape=(target_grid.size,)
                ),
            )
        return self.operator_cache[key]

    def tangential_grid_remap_operator(self, source_grid, target_grid):
        """Return a cached tangential grid-remap operator."""
        if source_grid.same_as(target_grid):
            return identity_linear_map((2, source_grid.size))
        matrix_key = self.remap_matrix_key(
            "tangential_grid_remap_matrix", source_grid, target_grid
        )
        key = ("tangential_grid_remap", matrix_key, bool(jax_enabled()))
        if key not in self.operator_cache:
            matrix = self._cached_remap_matrix(
                matrix_key,
                lambda: self.build_tangential_grid_remap_matrix(source_grid, target_grid),
            )
            self._store_operator(
                key,
                as_linear_map(
                    matrix,
                    input_shape=(2, source_grid.size),
                    output_shape=(2, target_grid.size),
                ),
            )
        return self.operator_cache[key]

    def interpolate_vector(
        self, u_theta, u_phi, u_radial, theta, phi, theta_target, phi_target, **kwargs
    ):
        """Interpolate canonical spherical vectors through CS panels."""
        basis = self.basis
        theta_target, phi_target = np.broadcast_arrays(theta_target, phi_target)
        target_shape = theta_target.shape
        xi, eta, block = basis.mesh.projection.geographic_to_cube(phi_target, 90 - theta_target)
        xi, eta, block = xi.reshape(-1), eta.reshape(-1), block.reshape(-1)

        theta, phi = np.broadcast_arrays(theta, phi)
        source_shape = theta.shape
        theta, phi = theta.reshape(-1), phi.reshape(-1)

        u_theta = np.asarray(u_theta)
        u_phi = np.asarray(u_phi)
        u_radial = np.asarray(u_radial)
        if u_theta.shape[: len(source_shape)] == source_shape:
            value_shape = u_theta.shape[len(source_shape) :]
            u_theta_values = u_theta.reshape((theta.size,) + value_shape)
            u_phi_values = u_phi.reshape((theta.size,) + value_shape)
            u_radial_values = u_radial.reshape((theta.size,) + value_shape)
        else:
            u_theta_values, u_phi_values, u_radial_values, theta_b, phi_b = np.broadcast_arrays(
                u_theta,
                u_phi,
                u_radial,
                theta.reshape(source_shape),
                phi.reshape(source_shape),
            )
            value_shape = ()
            u_theta_values = u_theta_values.reshape(-1)
            u_phi_values = u_phi_values.reshape(-1)
            u_radial_values = u_radial_values.reshape(-1)
            theta = theta_b.reshape(-1)
            phi = phi_b.reshape(-1)

        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        position = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        u_xi, u_eta, u_block = basis.mesh.projection.geographic_to_cube(phi, 90 - theta)
        geographic_to_panel = basis.mesh.projection.enu_to_cube_vector_matrix(
            u_xi, u_eta, radius=1, face=u_block
        )
        enu_values = np.stack([u_phi_values, -u_theta_values, u_radial_values], axis=1)
        panel_values = np.einsum("nij,nj...->ni...", geographic_to_panel, enu_values)

        interpolated_panel = np.empty((block.size, 3) + value_shape, dtype=np.float64)
        for block_index in range(6):
            panel_rotation = basis.mesh.projection.face_to_face_vector_matrix(
                u_xi, u_eta, u_block, block_index
            )
            values_on_panel = np.einsum("nij,nj...->ni...", panel_rotation, panel_values)

            _, panel_theta, panel_phi = basis.mesh.projection.cube_to_spherical(
                0, 0, block_index, degrees=False
            )
            panel_center = np.hstack(
                (
                    np.sin(panel_theta) * np.cos(panel_phi),
                    np.sin(panel_theta) * np.sin(panel_phi),
                    np.cos(panel_theta),
                )
            ).reshape((-1, 1))
            source_mask = np.sum(panel_center * position, axis=0) > 0
            source_xi, source_eta, _ = basis.mesh.projection.geographic_to_cube(
                phi, 90 - theta, face=block_index
            )
            target_mask = block == block_index
            interpolated_panel[target_mask] = griddata(
                np.column_stack((source_xi[source_mask], source_eta[source_mask])),
                values_on_panel[source_mask],
                np.column_stack((xi[target_mask], eta[target_mask])),
                **kwargs,
            )

        panel_to_geographic = basis.mesh.projection.cube_to_enu_vector_matrix(
            xi, eta, radius=1, face=block
        )
        interpolated_enu = np.einsum("nij,nj...->ni...", panel_to_geographic, interpolated_panel)
        return tuple(
            component.reshape(target_shape + value_shape)
            for component in (
                -interpolated_enu[:, 1],
                interpolated_enu[:, 0],
                interpolated_enu[:, 2],
            )
        )

    def interpolate_scalar(self, scalar, theta, phi, theta_target, phi_target, **kwargs):
        """Interpolate scalar values through CS panels."""
        basis = self.basis
        theta_target, phi_target = np.broadcast_arrays(theta_target, phi_target)
        target_shape = theta_target.shape
        xi, eta, block = basis.mesh.projection.geographic_to_cube(phi_target, 90 - theta_target)
        xi, eta, block = xi.reshape(-1), eta.reshape(-1), block.reshape(-1)

        theta, phi = np.broadcast_arrays(theta, phi)
        source_shape = theta.shape
        theta, phi = theta.reshape(-1), phi.reshape(-1)

        scalar = np.asarray(scalar)
        if scalar.shape[: len(source_shape)] == source_shape:
            value_shape = scalar.shape[len(source_shape) :]
            scalar_values = scalar.reshape((theta.size,) + value_shape)
        else:
            scalar_values, theta_broadcast, phi_broadcast = np.broadcast_arrays(
                scalar, theta.reshape(source_shape), phi.reshape(source_shape)
            )
            value_shape = ()
            scalar_values = scalar_values.reshape(-1)
            theta = theta_broadcast.reshape(-1)
            phi = phi_broadcast.reshape(-1)

        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        position = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
        interpolated = np.empty((block.size,) + value_shape, dtype=np.float64)

        for block_index in range(6):
            _, panel_theta, panel_phi = basis.mesh.projection.cube_to_spherical(
                0, 0, block_index, degrees=False
            )
            panel_center = np.hstack(
                (
                    np.sin(panel_theta) * np.cos(panel_phi),
                    np.sin(panel_theta) * np.sin(panel_phi),
                    np.cos(panel_theta),
                )
            ).reshape((-1, 1))
            source_mask = np.sum(panel_center * position, axis=0) > 0
            source_xi, source_eta, _ = basis.mesh.projection.geographic_to_cube(
                phi, 90 - theta, face=block_index
            )
            target_mask = block == block_index
            interpolated[target_mask] = griddata(
                np.column_stack((source_xi[source_mask], source_eta[source_mask])),
                scalar_values[source_mask],
                np.column_stack((xi[target_mask], eta[target_mask])),
                **kwargs,
            )

        return interpolated.reshape(target_shape + value_shape)


__all__ = ["GlobalCSMesh"]
