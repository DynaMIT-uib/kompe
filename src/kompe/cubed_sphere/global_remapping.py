"""Interpolation and cached remapping between global cubed-sphere grids."""

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from kompe.math import as_linear_map, identity_linear_map
from kompe.math.backend import backend_context, to_numpy


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

        # Delaunay triangulation and sparse assembly are SciPy CPU work.  The
        # cube-coordinate round trip must use the same NumPy arithmetic as the
        # triangulation, especially for points on face boundaries.
        with backend_context("numpy"):
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

    def _face_interpolation_stencils(self, theta, phi, theta_target, phi_target):
        """Return interpolation vertices and weights for each target face."""
        basis = self.basis
        xi_target, eta_target, target_face = basis.mesh.projection.geographic_to_cube(
            phi_target, 90 - theta_target
        )
        xi_target = xi_target.reshape(-1)
        eta_target = eta_target.reshape(-1)
        target_face = target_face.reshape(-1)

        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        r = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
        stencils = []

        for face_index in range(6):
            target_index = np.flatnonzero(target_face == face_index)
            if target_index.size == 0:
                continue

            _, th0, ph0 = basis.mesh.projection.cube_to_spherical(0, 0, face_index, degrees=False)
            r0 = np.array(
                [np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)]
            ).reshape((-1, 1))
            source_mask = np.sum(r0 * r, axis=0) > 0
            source_index = np.flatnonzero(source_mask)

            xi_source, eta_source, _ = basis.mesh.projection.geographic_to_cube(
                phi, 90 - theta, face=face_index
            )
            source_points = np.column_stack([xi_source[source_mask], eta_source[source_mask]])
            target_points = np.column_stack([xi_target[target_index], eta_target[target_index]])
            vertices, weights = self.linear_interpolation_weights(source_points, target_points)
            stencils.append((face_index, target_index, source_index[vertices], weights))

        return stencils

    def build_scalar_grid_remap_matrix(self, source_grid, target_grid):
        """Build a sparse scalar grid remap."""
        theta, phi = self.grid_theta_phi(source_grid)
        theta_target, phi_target = self.grid_theta_phi(target_grid)
        stencils = self._face_interpolation_stencils(theta, phi, theta_target, phi_target)

        rows = []
        cols = []
        data = []
        for _, target_index, source_vertices, weights in stencils:
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
        stencils = self._face_interpolation_stencils(theta, phi, theta_target, phi_target)

        xi_source, eta_source, source_face = basis.mesh.projection.geographic_to_cube(
            phi, 90 - theta
        )
        source_transform = basis.mesh.projection.enu_to_cube_vector_array(
            xi_source, eta_source, radius=1, face=source_face
        )

        xi_target, eta_target, target_face = basis.mesh.projection.geographic_to_cube(
            phi_target, 90 - theta_target
        )
        target_transform = basis.mesh.projection.cube_to_enu_vector_array(
            xi_target, eta_target, radius=1, face=target_face
        )

        n_source = theta.size
        n_target = theta_target.size
        out_components = np.arange(2)
        rows = []
        cols = []
        data = []

        for face_index, target_index, source_vertices, weights in stencils:
            qij = basis.mesh.projection.face_to_face_vector_array(
                xi_source, eta_source, source_face, face_index
            )
            source_to_face = np.einsum("nij,njk->nik", qij, source_transform)
            source_coeff = source_to_face[source_vertices]
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
        key = ("scalar_grid_remap", matrix_key)
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
        key = ("tangential_grid_remap", matrix_key)
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

    def _face_interpolation_points(self, theta, phi, xi_target, eta_target, face_target):
        """Yield source and target points for interpolation on each cube face."""
        th, ph = np.deg2rad(theta), np.deg2rad(phi)
        position = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

        for face in range(6):
            _, theta_center, phi_center = self.basis.mesh.projection.cube_to_spherical(
                0, 0, face, degrees=False
            )
            face_center = np.array(
                [
                    np.sin(theta_center) * np.cos(phi_center),
                    np.sin(theta_center) * np.sin(phi_center),
                    np.cos(theta_center),
                ]
            ).reshape(3, 1)
            source_mask = np.sum(face_center * position, axis=0) > 0
            xi_source, eta_source, _ = self.basis.mesh.projection.geographic_to_cube(
                phi, 90 - theta, face=face
            )
            target_mask = face_target == face
            yield (
                face,
                source_mask,
                target_mask,
                np.column_stack((xi_source[source_mask], eta_source[source_mask])),
                np.column_stack((xi_target[target_mask], eta_target[target_mask])),
            )

    def interpolate_vector(
        self, u_theta, u_phi, u_radial, theta, phi, theta_target, phi_target, **kwargs
    ):
        """Interpolate canonical spherical vectors through cube faces."""
        basis = self.basis
        theta_target, phi_target = np.broadcast_arrays(
            to_numpy(theta_target), to_numpy(phi_target)
        )
        target_shape = theta_target.shape
        xi, eta, target_face = basis.mesh.projection.geographic_to_cube(
            phi_target, 90 - theta_target
        )
        xi, eta, target_face = (
            xi.reshape(-1),
            eta.reshape(-1),
            target_face.reshape(-1),
        )

        theta, phi = np.broadcast_arrays(to_numpy(theta), to_numpy(phi))
        source_shape = theta.shape
        theta, phi = theta.reshape(-1), phi.reshape(-1)

        u_theta = np.asarray(to_numpy(u_theta))
        u_phi = np.asarray(to_numpy(u_phi))
        u_radial = np.asarray(to_numpy(u_radial))
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

        source_xi, source_eta, source_face = basis.mesh.projection.geographic_to_cube(
            phi, 90 - theta
        )
        geographic_to_face = basis.mesh.projection.enu_to_cube_vector_array(
            source_xi, source_eta, radius=1, face=source_face
        )
        enu_values = np.stack([u_phi_values, -u_theta_values, u_radial_values], axis=1)
        face_values = np.einsum("nij,nj...->ni...", geographic_to_face, enu_values)

        interpolated_face = np.empty((target_face.size, 3) + value_shape, dtype=np.float64)
        for (
            face_index,
            source_mask,
            target_mask,
            source_points,
            target_points,
        ) in self._face_interpolation_points(theta, phi, xi, eta, target_face):
            face_rotation = basis.mesh.projection.face_to_face_vector_array(
                source_xi, source_eta, source_face, face_index
            )
            values_on_face = np.einsum("nij,nj...->ni...", face_rotation, face_values)
            interpolated_face[target_mask] = griddata(
                source_points,
                values_on_face[source_mask],
                target_points,
                **kwargs,
            )

        face_to_geographic = basis.mesh.projection.cube_to_enu_vector_array(
            xi, eta, radius=1, face=target_face
        )
        interpolated_enu = np.einsum("nij,nj...->ni...", face_to_geographic, interpolated_face)
        return tuple(
            component.reshape(target_shape + value_shape)
            for component in (
                -interpolated_enu[:, 1],
                interpolated_enu[:, 0],
                interpolated_enu[:, 2],
            )
        )

    def interpolate_scalar(self, scalar, theta, phi, theta_target, phi_target, **kwargs):
        """Interpolate scalar values through cube faces."""
        basis = self.basis
        theta_target, phi_target = np.broadcast_arrays(
            to_numpy(theta_target), to_numpy(phi_target)
        )
        target_shape = theta_target.shape
        xi, eta, target_face = basis.mesh.projection.geographic_to_cube(
            phi_target, 90 - theta_target
        )
        xi, eta, target_face = (
            xi.reshape(-1),
            eta.reshape(-1),
            target_face.reshape(-1),
        )

        theta, phi = np.broadcast_arrays(to_numpy(theta), to_numpy(phi))
        source_shape = theta.shape
        theta, phi = theta.reshape(-1), phi.reshape(-1)

        scalar = np.asarray(to_numpy(scalar))
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

        interpolated = np.empty((target_face.size,) + value_shape, dtype=np.float64)

        for (
            _,
            source_mask,
            target_mask,
            source_points,
            target_points,
        ) in self._face_interpolation_points(theta, phi, xi, eta, target_face):
            interpolated[target_mask] = griddata(
                source_points,
                scalar_values[source_mask],
                target_points,
                **kwargs,
            )

        return interpolated.reshape(target_shape + value_shape)


__all__ = []
