"""Finite-difference operators for cubed-sphere grids."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import binom

from kompe.cubed_sphere import cs_coordinates
from kompe.cubed_sphere.finite_differences import finite_difference_weights


def _shift_rows_into_bounds(values, lower, upper):
    """Shift each row intact until it lies within inclusive bounds."""
    values = np.asarray(values)
    row_min = values.min(axis=1, keepdims=True)
    row_max = values.max(axis=1, keepdims=True)
    if np.any(row_max - row_min > upper - lower):
        raise ValueError("Row range is too large for the requested bounds.")
    return values - np.minimum(row_min, lower) + lower - np.maximum(row_max, upper) + upper


def global_cs_derivative_matrices(
    projection,
    cells_per_face,
    *,
    stencil_half_width=1,
    cross_face_points=4,
):
    """Return the native-grid ``(d/dxi, d/deta)`` matrices."""
    if stencil_half_width < 1:
        raise ValueError("stencil_half_width must be at least 1")

    shape = (6, cells_per_face, cells_per_face)
    size = np.prod(shape)
    h = cs_coordinates.face_coordinate(1, cells_per_face) - cs_coordinates.face_coordinate(
        0, cells_per_face
    )
    face, i, j = map(
        np.ravel,
        np.meshgrid(
            np.arange(6),
            np.arange(cells_per_face),
            np.arange(cells_per_face),
            indexing="ij",
        ),
    )

    offsets = np.hstack((np.r_[-stencil_half_width:0], np.r_[1 : stencil_half_width + 1]))
    offset_count = len(offsets)
    weights = np.repeat(finite_difference_weights(offsets, order=1, h=h), size)
    face_repeated = np.tile(face, offset_count)
    i_repeated = np.tile(i, offset_count)
    j_repeated = np.tile(j, offset_count)
    rows = np.tile(np.ravel_multi_index((face, i, j), shape), offset_count)

    dxi = _cross_face_interpolation_matrix(
        projection,
        face_repeated,
        np.hstack([i + offset for offset in offsets]),
        j_repeated,
        cells_per_face,
        cross_face_points,
        rows=rows,
        weights=weights,
    )
    deta = _cross_face_interpolation_matrix(
        projection,
        face_repeated,
        i_repeated,
        np.hstack([j + offset for offset in offsets]),
        cells_per_face,
        cross_face_points,
        rows=rows,
        weights=weights,
    )
    return dxi, deta


def _cross_face_interpolation_matrix(
    projection,
    face,
    i,
    j,
    cells_per_face,
    point_count,
    *,
    weights=None,
    rows=None,
):
    """Interpolate off-face stencil points from the adjoining face."""
    if point_count > cells_per_face:
        raise ValueError("cross_face_points cannot exceed cells_per_face")

    face, i, j = map(np.ravel, [face, i, j])
    shape = (6, cells_per_face, cells_per_face)
    size = np.prod(shape)

    if rows is None:
        rows = np.arange(face.size)
    if weights is None:
        weights = np.ones(face.size)
    weights = weights / point_count

    h = cs_coordinates.face_coordinate(1, cells_per_face) - cs_coordinates.face_coordinate(
        0, cells_per_face
    )
    columns = np.full(face.size, -1, dtype=np.int64)

    xi = cs_coordinates.face_coordinate(i + 0.5, cells_per_face)
    eta = cs_coordinates.face_coordinate(j + 0.5, cells_per_face)
    _, theta, phi = projection.cube_to_spherical(xi, eta, face, radius=1.0, degrees=True)
    new_xi, new_eta, new_face = projection.geographic_to_cube(phi, 90 - theta)
    new_i = new_xi / h + (cells_per_face - 1) / 2
    new_j = new_eta / h + (cells_per_face - 1) / 2

    on_i_grid_line = np.isclose(new_i - np.rint(new_i), 0)
    on_j_grid_line = np.isclose(new_j - np.rint(new_j), 0)
    if not np.all(on_i_grid_line | on_j_grid_line):
        raise RuntimeError(
            "Cross-face interpolation points must align with at least one target-grid axis."
        )

    integer_pairs = on_i_grid_line & on_j_grid_line
    columns[integer_pairs] = np.ravel_multi_index(
        (
            new_face[integer_pairs],
            np.rint(new_i[integer_pairs]).astype(np.int64),
            np.rint(new_j[integer_pairs]).astype(np.int64),
        ),
        shape,
    )

    interpolate_i = ~on_i_grid_line
    interpolate_j = ~on_j_grid_line
    if np.any(interpolate_i & interpolate_j):
        raise RuntimeError(
            "Cross-face interpolation cannot interpolate along both grid axes at once."
        )
    if np.count_nonzero(interpolate_i | interpolate_j) != np.count_nonzero(columns == -1):
        raise RuntimeError(
            "Cross-face interpolation classification is inconsistent with target columns."
        )

    fractional_i = new_i[interpolate_i].reshape((-1, 1))
    fractional_j = new_j[interpolate_j].reshape((-1, 1))
    points = np.arange(point_count).reshape((1, -1))
    i_points = _shift_rows_into_bounds(
        points + np.int64(np.ceil(fractional_i)) - point_count // 2,
        0,
        cells_per_face - 1,
    )
    j_points = _shift_rows_into_bounds(
        points + np.int64(np.ceil(fractional_j)) - point_count // 2,
        0,
        cells_per_face - 1,
    )

    i_distances = fractional_i - i_points
    j_distances = fractional_j - j_points
    barycentric_weights = (-1) ** points * binom(point_count - 1, points)
    i_weights = barycentric_weights / i_distances
    i_weights /= np.sum(i_weights, axis=1).reshape((-1, 1))
    j_weights = barycentric_weights / j_distances
    j_weights /= np.sum(j_weights, axis=1).reshape((-1, 1))

    stacked_weights = np.tile(weights, (point_count, 1)).T
    stacked_columns = np.tile(columns, (point_count, 1)).T
    stacked_rows = np.tile(rows, (point_count, 1)).T

    stacked_columns[interpolate_i] = np.ravel_multi_index(
        (
            np.tile(new_face[interpolate_i], (point_count, 1)).T,
            i_points,
            np.rint(np.tile(new_j[interpolate_i], (point_count, 1))).astype(np.int64).T,
        ),
        shape,
    )
    stacked_columns[interpolate_j] = np.ravel_multi_index(
        (
            np.tile(new_face[interpolate_j], (point_count, 1)).T,
            np.rint(np.tile(new_i[interpolate_j], (point_count, 1))).astype(np.int64).T,
            j_points,
        ),
        shape,
    )
    stacked_weights[interpolate_i] *= i_weights * point_count
    stacked_weights[interpolate_j] *= j_weights * point_count

    matrix = coo_matrix(
        (
            stacked_weights.reshape(-1),
            (stacked_rows.reshape(-1), stacked_columns.reshape(-1)),
        ),
        shape=(rows.max() + 1, size),
    )
    matrix.sum_duplicates()
    return matrix


__all__ = ["global_cs_derivative_matrices"]
