"""Coordinate projection for a rotated regional cubed-sphere face."""

from pathlib import Path

import numpy as np

from kompe.cubed_sphere import cs_coordinates, cs_vectors
from kompe.spherical import ecef_to_enu, enu_to_ecef

_DATA_PATH = Path(__file__).resolve().parents[1] / "data"
_NORTH_FACE = 4


def _rotate_spherical_coordinates(lon, lat, rotation):
    """Rotate spherical coordinates with a Cartesian rotation matrix."""
    lon, lat = np.broadcast_arrays(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    shape = lon.shape
    lon = np.deg2rad(lon.reshape(-1))
    lat = np.deg2rad(lat.reshape(-1))
    xyz = np.column_stack(
        (
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        )
    )
    rotated = np.einsum("ij,nj->ni", rotation, xyz)
    rotated_lon = np.rad2deg(np.arctan2(rotated[:, 1], rotated[:, 0]))
    rotated_lat = np.rad2deg(np.arctan2(rotated[:, 2], np.hypot(rotated[:, 0], rotated[:, 1])))
    return rotated_lon.reshape(shape), rotated_lat.reshape(shape)


class RegionalCSProjection:
    """Rotated north-face cubed-sphere coordinate chart."""

    def __init__(self, position, orientation):
        """Set up a regional cubed-sphere chart.

        The RegionalCSProjection is set up by
        1) rotating to a local coordinate system in which 'position'
        is at the pole, and 'orientation' defines the x axis (prime meridian)
        2) applying the Ronchi et al. conversions to xi, eta coords on the
        local coordinates

        Parameters
        ----------
        position : array-like of (longitude, latitude)
            Centre at which the cube surface is tangential to the sphere,
            in degrees. The tuple order is explicitly longitude first,
            latitude second.
        orientation: scalar or 2-element array-like
            orientation of the cube surface.
            if scalar: angle in degrees that defines the xi axis: orientation = 0 / 180
            implies a xi axis in the east-west direction, positive towards east / west.
            orientation = 90 / 270 implies a xi axis towards north / south.
            if 2-element array-like: The elements denote the eastward and northward components
            of a vector that is aligned with the xi axis.
        """
        self.position = np.asarray(position, dtype=float)
        if self.position.shape != (2,) or not np.isfinite(self.position).all():
            raise ValueError("position must contain finite longitude and latitude values")
        if not -90.0 <= self.position[1] <= 90.0:
            raise ValueError("position latitude must be between -90 and 90 degrees")

        self.orientation = np.asarray(orientation, dtype=float)
        if not np.isfinite(self.orientation).all():
            raise ValueError("orientation must contain finite values")

        if self.orientation.size == 2:  # interpreted as a east, north component:
            orientation_norm = np.linalg.norm(self.orientation)
            if orientation_norm == 0:
                raise ValueError("orientation must be non-zero")
            if not np.isclose(orientation_norm, 1.0, rtol=0.0, atol=1e-15):
                self.orientation = self.orientation / orientation_norm
        else:  # interpreted as scalar
            if self.orientation.size != 1:
                raise ValueError("orientation must be either scalar or have 2 elements")
            angle = float(self.orientation.reshape(-1)[0])
            angle = np.deg2rad(angle)
            self.orientation = np.array([np.cos(angle), np.sin(angle)])
        orientation_enu = np.array([self.orientation[0], self.orientation[1], 0]).reshape((1, 3))

        self.lon0, self.lat0 = self.position
        longitude = np.deg2rad(self.lon0)
        latitude = np.deg2rad(self.lat0)

        # Local radial direction, expressed in geographic ECEF coordinates.
        self.local_z_axis = np.array(
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ]
        )

        # On the north face, increasing xi follows the local Cartesian y axis.
        self.local_y_axis = enu_to_ecef(
            orientation_enu,
            np.array(self.lat0),
            np.array(self.lon0),
        ).flatten()

        # Complete the right-handed local Cartesian frame.
        self.local_x_axis = np.cross(self.local_y_axis, self.local_z_axis)

        # define rotation matrices for rotations between local and geocentric:
        self.geographic_to_local_matrix = np.vstack(
            (self.local_x_axis, self.local_y_axis, self.local_z_axis)
        )  # rotation matrix from GEO to rotated coords (ECEF)
        self.local_to_geographic_matrix = self.geographic_to_local_matrix.T
        for name in (
            "position",
            "orientation",
            "local_x_axis",
            "local_y_axis",
            "local_z_axis",
            "geographic_to_local_matrix",
            "local_to_geographic_matrix",
        ):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            setattr(self, name, values)

    def __repr__(self):
        """Summarize the regional chart for interactive inspection."""
        return (
            f"RegionalCSProjection(position={tuple(map(float, self.position))}, "
            f"orientation={tuple(map(float, self.orientation))})"
        )

    @property
    def signature(self):
        """Return immutable projection identity for grids and caches."""
        return (
            "REGIONAL_CS_PROJECTION",
            tuple(float(value) for value in self.position),
            tuple(float(value) for value in self.orientation),
        )

    def geographic_to_cube(self, lon, lat):
        """Convert geocentric coordinates to cube coordinates ``(xi, eta)``.

        Inputs are broadcast together and output has the resulting shape.
        Points on the opposite hemisphere of the local projection are NaN.

        Parameters
        ----------
        lon: array
            geocentric longitude(s) [deg] to convert to cube coords
        lat: array:
            geocentric latitude(s) [deg] to convert to cube coords.

        Returns
        -------
        xi: array
            xi, as defined in Ronchi et al, after lon, lat have been
            converted to local coordinates. Unit is radians.
        eta: array
            eta, as defined in Ronchi et al., after lon, lat have been
            converted to local coordinates. Unit is radians.

        """
        lon, lat = np.broadcast_arrays(np.asarray(lon), np.asarray(lat))
        local_lon, local_lat = self.geographic_to_local(lon, lat)
        xi, eta, _ = cs_coordinates.geographic_to_cube(
            local_lon,
            local_lat,
            block=_NORTH_FACE,
        )

        on_local_hemisphere = local_lat >= 0
        return (
            np.where(on_local_hemisphere, xi, np.nan),
            np.where(on_local_hemisphere, eta, np.nan),
        )

    def cube_to_geographic(self, xi, eta):
        """Convert cube coordinates ``(xi, eta)`` to geocentric ``(lon, lat)``.

        Inputs are broadcast together and output has the resulting shape.

        Parameters
        ----------
        xi: array
            Cubed-sphere xi coordinate(s) [rad].
        eta: array
            Cubed-sphere eta coordinate(s) [rad].

        Returns
        -------
        lon: array
            Geocentric longitude(s) [deg].
        lat: array
            Geocentric latitude(s) [deg].


        """
        xi, eta = np.broadcast_arrays(np.asarray(xi, dtype=float), np.asarray(eta, dtype=float))
        _, theta, phi = cs_coordinates.cube_to_spherical(
            xi,
            eta,
            block=_NORTH_FACE,
            deg=True,
        )
        return self.local_to_geographic(phi, 90 - theta)

    def geographic_to_local(self, lon, lat):
        """Convert geocentric coordinates to the rotated local coordinates.

        Inputs are broadcast together and output has the resulting shape.

        Parameters
        ----------
        lon: array-like
            array of longitudes [deg]
        lat: array-like
            array of latitudes [deg]

        Returns
        -------
        lon: array-like
            array of longitudes [deg] in new coordinate system
        lat: array-like
            array of latitudes [deg] in new coordinate system
        """
        return _rotate_spherical_coordinates(lon, lat, self.geographic_to_local_matrix)

    def local_to_geographic(self, lon, lat):
        """Convert rotated local coordinates to geocentric coordinates.

        Inputs are broadcast together and output has the resulting shape.

        Parameters
        ----------
        lon: array-like
            array of longitudes [deg]
        lat: array-like
            array of latitudes [deg]

        Returns
        -------
        lon: array-like
            array of longitudes [deg] in new coordinate system
        lat: array-like
            array of latitudes [deg] in new coordinate system

        """
        return _rotate_spherical_coordinates(lon, lat, self.local_to_geographic_matrix)

    def local_to_geographic_enu_rotation(self, lon, lat):
        """Return rotation matrices from local ENU to geocentric ENU.

        Parameters
        ----------
        lon: array-like
            array of longitudes (local coords) for which rotation matrices should be calculated
        lat: array-like
            array of latitudes (local coords) for which rotation matrices should be calculated

        Returns
        -------
        R_localenu2geoenu: array
            Rotation matrices that rotate ENU vectors in local coordinates to ENU vectors
            in geocentric coordinates. Shape is (N, 2, 2). To get the opposite rotation,
            use the transpose by swapping the last two axes of the array. The rotation
            matrices are (2, 2), and should be applied on (east, north) components. The
            upward component is the same in the two coordinate systems.
            N is the size of lon and lat (they will be flattened)
        """
        lon, lat = map(np.ravel, np.broadcast_arrays(lon, lat))
        geographic_lon, geographic_lat = self.local_to_geographic(lon, lat)
        local_east = enu_to_ecef(np.tile((1.0, 0.0, 0.0), (lon.size, 1)), lat, lon)
        local_north = enu_to_ecef(np.tile((0.0, 1.0, 0.0), (lon.size, 1)), lat, lon)
        geographic_east = ecef_to_enu(
            np.einsum("ij,nj->ni", self.local_to_geographic_matrix, local_east),
            geographic_lat,
            geographic_lon,
        )[:, :2]
        geographic_north = ecef_to_enu(
            np.einsum("ij,nj->ni", self.local_to_geographic_matrix, local_north),
            geographic_lat,
            geographic_lon,
        )[:, :2]
        return np.stack((geographic_east, geographic_north), axis=2)

    def geographic_vector_to_cube(self, east, north, lon, lat):
        """Project geographic tangent vectors into cube-coordinate components.

        Parameters
        ----------
        east: array-like
            Array of N eastward (geo) components
        north: array-like
            Array of N northward (geo) components
        lon: array-like
            Array of N longitudes that represent vector positions
        lat: array-like
            Array of N latitudes that represent vector positions

        Returns
        -------
        xi: array-like
            N element array of xi coordinates
        eta: array-like
            N element array of eta coordinates
        Axi: array-like
            N element array of vector components in xi direction
        Aeta: array-like
            N element array of vector components in eta direction

        """
        east, north, lon, lat = map(
            np.ravel,
            np.broadcast_arrays(east, north, lon, lat),
        )
        xi, eta = self.geographic_to_cube(lon, lat)
        geographic_ecef = enu_to_ecef(
            np.column_stack((east, north, np.zeros_like(east))),
            lat,
            lon,
        )
        local_ecef = np.einsum("ij,nj->ni", self.geographic_to_local_matrix, geographic_ecef)
        cube_matrix = cs_vectors._cartesian_to_cube_matrix(
            xi,
            eta,
            r=1.0,
            block=_NORTH_FACE,
        )
        cube = np.einsum("nij,nj->ni", cube_matrix, local_ecef)
        return xi, eta, cube[:, 0], cube[:, 1]

    def cube_vector_to_geographic(self, Axi, Aeta, xi, eta):
        """Convert cube-coordinate tangent components to geographic ENU.

        Parameters
        ----------
        Axi: array-like
            Array of N xi components
        Aeta: array-like
            Array of N eta components
        xi: array-like
            Array of N xi coords that represent vector positions
        eta: array-like
            Array of N eta coords that represent vector positions

        Returns
        -------
        lon: array-like
            N element array of lon coordinates
        lat: array-like
            N element array of lat coordinates
        east: array-like
            N element array of vector components in east direction
        north: array-like
            N element array of vector components in north direction

        """
        Axi, Aeta, xi, eta = map(
            np.ravel,
            np.broadcast_arrays(Axi, Aeta, xi, eta),
        )
        lon, lat = self.cube_to_geographic(xi, eta)
        cube = np.column_stack((Axi, Aeta, np.zeros_like(Axi)))
        cartesian_matrix = cs_vectors._cube_to_cartesian_matrix(
            xi,
            eta,
            r=1.0,
            block=_NORTH_FACE,
        )
        local_ecef = np.einsum("nij,nj->ni", cartesian_matrix, cube)
        geographic_ecef = np.einsum("ij,nj->ni", self.local_to_geographic_matrix, local_ecef)
        geographic = ecef_to_enu(geographic_ecef, lat, lon)
        return lon, lat, geographic[:, 0], geographic[:, 1]

    def projected_coastlines(self, resolution="50m"):
        """Generate coastlines in projected coordinates."""
        coastlines = np.load(_DATA_PATH / f"coastlines_{resolution}.npz")
        for key in coastlines:
            lat, lon = coastlines[key]
            yield self.geographic_to_cube(lon, lat)

    def differential_elements(self, xi, eta, dxi, deta, radius=1.0):
        """Calculate magnitudes of line and surface elements.

        Implementation of equations 18-20 of Ronchi et al.

        Broadcasting rules apply, so that output will have the shape of
        the combination of input parameters:
        dS.shape will be equal to (xi * eta * dxi * deta).shape

        xi, eta, dxi, deta must all be given in radians. dlxi and dleta
        will be given in units of R, and dS in units of R squared (default
        is radian and steradian)

        Parameters
        ----------
        xi: array-like
            xi coordinate(s) of surface element(s)
        eta: array-like
            eta coordinate(s) of surface element(s)
        dxi: array-like
            dimension(s) of surface element(s) in xi direction
        deta: array-like
            dimension(s) of surface element(s) in eta direction
        radius: float, optional
            radius of the sphere - default is 1

        Returns
        -------
        dlxi: array-like
            Length of line element(s), in radians or in units of ``radius``,
            along xi direction
        dleta: array-like
            Length of line element(s), in radians or in units of ``radius``,
            along eta direction
        dS: array-like
            Area(s) of surface element(s), in steradians or in
            squared units of ``radius``
        """
        xi, eta, dxi, deta, radius = np.broadcast_arrays(xi, eta, dxi, deta, radius)
        metric = cs_coordinates.surface_metric_tensor(xi, eta, r=radius).reshape(xi.shape + (2, 2))

        dlxi = np.sqrt(metric[..., 0, 0]) * dxi
        dleta = np.sqrt(metric[..., 1, 1]) * deta
        area_scale = np.sqrt(metric[..., 0, 0] * metric[..., 1, 1] - metric[..., 0, 1] ** 2)
        dS = area_scale * dxi * deta

        return dlxi, dleta, dS


__all__ = ["RegionalCSProjection"]
