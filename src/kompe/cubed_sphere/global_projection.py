"""Stateless global cubed-sphere coordinate and component transforms."""

from __future__ import annotations

from kompe.cubed_sphere import cs_coordinates, cs_vectors


class GlobalCSProjection:
    """Continuous coordinate chart for the six-face global cubed sphere.

    The global chart has no configuration state: the selected face is carried
    by each coordinate value.  The object provides the same projection
    collaborator boundary used by global and regional cubed-sphere meshes,
    while delegating the numerical implementation to ``cs_coordinates`` and
    ``cs_vectors``.
    """

    __slots__ = ()

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
        return cs_vectors.pc(
            xi,
            eta,
            r=radius,
            block=face,
            inverse=False,
        )

    @staticmethod
    def cube_to_cartesian_vector_matrix(xi, eta, radius=1, face=0):
        """Return cube-to-Cartesian component transformation matrices."""
        return cs_vectors.pc(
            xi,
            eta,
            r=radius,
            block=face,
            inverse=True,
        )

    @staticmethod
    def spherical_to_cube_vector_matrix(xi, eta, radius=1, face=0):
        """Return spherical-to-cube component transformation matrices."""
        return cs_vectors.ps(
            xi,
            eta,
            r=radius,
            block=face,
            inverse=False,
        )

    @staticmethod
    def cube_to_spherical_vector_matrix(xi, eta, radius=1, face=0):
        """Return cube-to-spherical component transformation matrices."""
        return cs_vectors.ps(
            xi,
            eta,
            r=radius,
            block=face,
            inverse=True,
        )

    @staticmethod
    def face_to_face_vector_matrix(xi, eta, source_face, target_face):
        """Return component transformations between global cube faces."""
        return cs_vectors.q_between_blocks(
            xi,
            eta,
            source_face,
            target_face,
        )

    @staticmethod
    def spherical_normalization_matrix(latitude, radius, inverse=False):
        """Return normalized/coordinate spherical component transforms."""
        return cs_vectors.spherical_q(
            latitude,
            radius,
            inverse=inverse,
        )


__all__ = ["GlobalCSProjection"]
