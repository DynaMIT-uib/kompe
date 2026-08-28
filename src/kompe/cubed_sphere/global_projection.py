"""Stateless global cubed-sphere coordinate and component transforms."""

from __future__ import annotations

from kompe.cubed_sphere import cs_coordinates, cs_vectors
from kompe.cubed_sphere.geometry_linalg import inverse_3x3
from kompe.math.backend import get_array_module


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
        return cs_coordinates.metric_delta(xi, eta)

    @staticmethod
    def metric_tensor(xi, eta, radius=1, covariant=True):
        """Return cubed-sphere metric tensors."""
        return cs_coordinates.metric_tensor(
            xi,
            eta,
            radius=radius,
            covariant=covariant,
        )

    @staticmethod
    def face_index(longitude, latitude):
        """Return global cubed-sphere face indices."""
        return cs_coordinates.face_index(longitude, latitude)

    @staticmethod
    def geographic_to_cube(longitude, latitude, face=None):
        """Map geographic longitude/latitude to global cube coordinates."""
        return cs_coordinates.geographic_to_cube(
            longitude,
            latitude,
            face=face,
        )

    @staticmethod
    def cube_to_cartesian(xi, eta, radius=1, face=0):
        """Map global cube coordinates to Cartesian coordinates."""
        return cs_coordinates.cube_to_cartesian(
            xi,
            eta,
            radius=radius,
            face=face,
        )

    @staticmethod
    def cube_to_spherical(xi, eta, face=0, radius=1, degrees=False):
        """Map global cube coordinates to spherical coordinates."""
        return cs_coordinates.cube_to_spherical(
            xi,
            eta,
            face,
            radius=radius,
            degrees=degrees,
        )

    @staticmethod
    def cartesian_to_cube_vector_matrix(xi, eta, radius=1, face=0):
        """Return Cartesian-to-cube component transformation matrices."""
        return cs_vectors._cartesian_to_cube_matrix(
            xi,
            eta,
            radius=radius,
            face=face,
        )

    @staticmethod
    def cube_to_cartesian_vector_matrix(xi, eta, radius=1, face=0):
        """Return cube-to-Cartesian component transformation matrices."""
        return cs_vectors._cube_to_cartesian_matrix(
            xi,
            eta,
            radius=radius,
            face=face,
        )

    @staticmethod
    def enu_to_cube_vector_matrix(xi, eta, radius=1, face=0):
        """Return ENU-to-``(xi, eta, radial)`` component matrices."""
        cube_to_cartesian = cs_vectors._cube_to_cartesian_matrix(
            xi,
            eta,
            radius=radius,
            face=face,
        )
        cartesian_to_cube = inverse_3x3(cube_to_cartesian)
        enu_to_cartesian = cs_vectors._enu_to_cartesian_matrix(cube_to_cartesian[:, :, 2])
        xp = get_array_module(cartesian_to_cube, enu_to_cartesian)
        return xp.einsum("nij,njk->nik", cartesian_to_cube, enu_to_cartesian)

    @staticmethod
    def cube_to_enu_vector_matrix(xi, eta, radius=1, face=0):
        """Return ``(xi, eta, radial)``-to-ENU component matrices."""
        cube_to_cartesian = cs_vectors._cube_to_cartesian_matrix(
            xi,
            eta,
            radius=radius,
            face=face,
        )
        enu_to_cartesian = cs_vectors._enu_to_cartesian_matrix(cube_to_cartesian[:, :, 2])
        xp = get_array_module(enu_to_cartesian, cube_to_cartesian)
        cartesian_to_enu = xp.swapaxes(enu_to_cartesian, 1, 2)
        return xp.einsum("nij,njk->nik", cartesian_to_enu, cube_to_cartesian)

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
