"""Stateless global cubed-sphere coordinate and component transforms."""

from __future__ import annotations

import numpy as np

from kompe.cubed_sphere import cs_coordinates, cs_vectors


class GlobalCSProjection:
    """Continuous coordinate chart for the six-face global cubed sphere.

    The global chart has no configuration state: the selected face is carried
    by each coordinate value. Regional projections rotate geographic
    coordinates into this chart's north face. Both projections use the same
    numerical implementation in ``cs_coordinates`` and ``cs_vectors``; their
    meshes retain separate regional-boundary and global-face topology.
    """

    __slots__ = ()

    def __repr__(self):
        """Return the stateless chart name."""
        return "GlobalCSProjection()"

    @property
    def signature(self):
        """Return the stable identity of the stateless global chart."""
        return ("GLOBAL_CS_PROJECTION",)

    @staticmethod
    def metric_delta(xi, eta):
        """Return the cubed-sphere metric delta parameter."""
        return cs_coordinates.delta(xi, eta)

    @staticmethod
    def metric_tensor(xi, eta, radius=1, covariant=True):
        """Return cubed-sphere metric tensors."""
        return cs_coordinates.metric_tensor(
            xi,
            eta,
            r=radius,
            covariant=covariant,
        )

    @staticmethod
    def face_index(longitude, latitude):
        """Return global cubed-sphere face indices."""
        return cs_coordinates.cube_face(longitude, latitude)

    @staticmethod
    def geographic_to_cube(longitude, latitude, face=None):
        """Map geographic longitude/latitude to global cube coordinates."""
        return cs_coordinates.geo_to_cube(
            longitude,
            latitude,
            block=face,
        )

    @staticmethod
    def cube_to_cartesian(xi, eta, radius=1, face=0):
        """Map global cube coordinates to Cartesian coordinates."""
        return cs_coordinates.cube_to_cartesian(
            xi,
            eta,
            r=radius,
            block=face,
        )

    @staticmethod
    def cube_to_spherical(xi, eta, face=0, radius=1, degrees=False):
        """Map global cube coordinates to spherical coordinates."""
        return cs_coordinates.cube_to_spherical(
            xi,
            eta,
            face,
            r=radius,
            deg=degrees,
        )

    @staticmethod
    def cartesian_to_cube_vector_matrix(xi, eta, radius=1, face=0):
        """Return Cartesian-to-cube component transformation matrices."""
        return cs_vectors._cartesian_to_cube_matrix(
            xi,
            eta,
            r=radius,
            block=face,
        )

    @staticmethod
    def cube_to_cartesian_vector_matrix(xi, eta, radius=1, face=0):
        """Return cube-to-Cartesian component transformation matrices."""
        return cs_vectors._cube_to_cartesian_matrix(
            xi,
            eta,
            r=radius,
            block=face,
        )

    @staticmethod
    def enu_to_cube_vector_matrix(xi, eta, radius=1, face=0):
        """Return ENU-to-``(xi, eta, radial)`` component matrices."""
        _, theta, _ = cs_coordinates.cube_to_spherical(
            xi,
            eta,
            face,
            r=radius,
            deg=True,
        )
        coordinate_to_cube = cs_vectors._spherical_coordinate_to_cube_matrix(
            xi,
            eta,
            r=radius,
            block=face,
        )
        enu_to_coordinate = cs_vectors._enu_to_spherical_coordinate_matrix(
            90.0 - theta,
            radius,
        )
        return np.einsum("nij,njk->nik", coordinate_to_cube, enu_to_coordinate)

    @staticmethod
    def cube_to_enu_vector_matrix(xi, eta, radius=1, face=0):
        """Return ``(xi, eta, radial)``-to-ENU component matrices."""
        _, theta, _ = cs_coordinates.cube_to_spherical(
            xi,
            eta,
            face,
            r=radius,
            deg=True,
        )
        cube_to_coordinate = cs_vectors._cube_to_spherical_coordinate_matrix(
            xi,
            eta,
            r=radius,
            block=face,
        )
        coordinate_to_enu = cs_vectors._spherical_coordinate_to_enu_matrix(
            90.0 - theta,
            radius,
        )
        return np.einsum("nij,njk->nik", coordinate_to_enu, cube_to_coordinate)

    @staticmethod
    def face_to_face_vector_matrix(xi, eta, source_face, target_face):
        """Return component transformations between global cube faces."""
        return cs_vectors._face_to_face_matrix(
            xi,
            eta,
            source_face,
            target_face,
        )


__all__ = ["GlobalCSProjection"]
