"""Coordinate projection for a rotated regional cubed-sphere face."""

import numpy as np

from kompe.cubed_sphere import cs_coordinates, cs_vectors
from kompe.math.backend import backend_context, get_array_module, to_numpy
from kompe.spherical import ecef_to_enu, enu_to_ecef, rotate_spherical_by_matrix

_NORTH_FACE = 4


class RegionalCSProjection:
    """Rotated north-face cubed-sphere coordinate chart."""

    def __init__(self, position, orientation):
        """Set up a regional cubed-sphere chart.

        The chart first rotates ``position`` to the north pole, with increasing
        xi aligned with ``orientation``, and then applies the Ronchi et al.
        north-face transformation to the local coordinates.

        Parameters
        ----------
        position : array-like of (longitude, latitude)
            Centre at which the cube surface is tangential to the sphere,
            in degrees. The tuple order is explicitly longitude first,
            latitude second.
        orientation : scalar or 2-element array-like
            Direction of increasing xi at ``position``. A scalar is an angle
            in degrees: 0 points east, 90 north, 180 west, and 270 south. A
            two-element value gives the eastward and northward components.
        """
        self.position = np.asarray(position, dtype=float)
        if self.position.shape != (2,) or not np.isfinite(self.position).all():
            raise ValueError("position must contain finite longitude and latitude values")
        if not -90.0 <= self.position[1] <= 90.0:
            raise ValueError("position latitude must be between -90 and 90 degrees")

        self.orientation = np.asarray(orientation, dtype=float)
        if not np.isfinite(self.orientation).all():
            raise ValueError("orientation must contain finite values")

        if self.orientation.size == 2:  # Eastward and northward components.
            orientation_norm = np.linalg.norm(self.orientation)
            if orientation_norm == 0:
                raise ValueError("orientation must be non-zero")
            if not np.isclose(orientation_norm, 1.0, rtol=0.0, atol=1e-15):
                self.orientation = self.orientation / orientation_norm
        else:  # Angle in degrees.
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
        with backend_context("numpy"):
            self.local_y_axis = to_numpy(
                enu_to_ecef(
                    orientation_enu,
                    np.array(self.lat0),
                    np.array(self.lon0),
                )
            ).flatten()

        # Complete the right-handed local Cartesian frame.
        self.local_x_axis = np.cross(self.local_y_axis, self.local_z_axis)

        # Rotation matrices between local and geocentric Cartesian coordinates.
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
        lon : array
            geocentric longitude(s) [deg] to convert to cube coords
        lat : array
            geocentric latitude(s) [deg] to convert to cube coords.

        Returns
        -------
        xi : array
            xi, as defined in Ronchi et al, after lon, lat have been
            converted to local coordinates. Unit is radians.
        eta : array
            eta, as defined in Ronchi et al., after lon, lat have been
            converted to local coordinates. Unit is radians.

        """
        xp = get_array_module(lon, lat)
        lon, lat = xp.broadcast_arrays(xp.asarray(lon), xp.asarray(lat))
        local_lon, local_lat = self.geographic_to_local(lon, lat)
        xi, eta, _ = cs_coordinates.geographic_to_cube(
            local_lon,
            local_lat,
            face=_NORTH_FACE,
        )

        on_local_hemisphere = local_lat >= 0
        return (
            xp.where(on_local_hemisphere, xi, xp.nan),
            xp.where(on_local_hemisphere, eta, xp.nan),
        )

    def cube_to_geographic(self, xi, eta):
        """Convert cube coordinates ``(xi, eta)`` to geocentric ``(lon, lat)``.

        Inputs are broadcast together and output has the resulting shape.

        Parameters
        ----------
        xi : array
            Cubed-sphere xi coordinate(s) [rad].
        eta : array
            Cubed-sphere eta coordinate(s) [rad].

        Returns
        -------
        lon : array
            Geocentric longitude(s) [deg].
        lat : array
            Geocentric latitude(s) [deg].


        """
        xp = get_array_module(xi, eta)
        xi, eta = xp.broadcast_arrays(xp.asarray(xi), xp.asarray(eta))
        _, theta, phi = cs_coordinates.cube_to_spherical(
            xi,
            eta,
            face=_NORTH_FACE,
            degrees=True,
        )
        return self.local_to_geographic(phi, 90 - theta)

    def geographic_to_local(self, lon, lat):
        """Convert geocentric coordinates to the rotated local coordinates.

        Inputs are broadcast together and output has the resulting shape.

        Parameters
        ----------
        lon : array-like
            Geocentric longitude [deg].
        lat : array-like
            Geocentric latitude [deg].

        Returns
        -------
        lon : array-like
            Longitude in the rotated coordinate system [deg].
        lat : array-like
            Latitude in the rotated coordinate system [deg].
        """
        local_lat, local_lon = rotate_spherical_by_matrix(
            lat, lon, self.geographic_to_local_matrix
        )
        return local_lon, local_lat

    def local_to_geographic(self, lon, lat):
        """Convert rotated local coordinates to geocentric coordinates.

        Inputs are broadcast together and output has the resulting shape.

        Parameters
        ----------
        lon : array-like
            Longitude in the rotated coordinate system [deg].
        lat : array-like
            Latitude in the rotated coordinate system [deg].

        Returns
        -------
        lon : array-like
            Geocentric longitude [deg].
        lat : array-like
            Geocentric latitude [deg].

        """
        geographic_lat, geographic_lon = rotate_spherical_by_matrix(
            lat, lon, self.local_to_geographic_matrix
        )
        return geographic_lon, geographic_lat

    def local_to_geographic_enu_rotation(self, lon, lat):
        """Return rotation matrices from local ENU to geocentric ENU.

        Parameters
        ----------
        lon : array-like
            array of longitudes (local coords) for which rotation matrices should be calculated
        lat : array-like
            array of latitudes (local coords) for which rotation matrices should be calculated

        Returns
        -------
        R_localenu2geoenu : array
            Rotation matrices that rotate ENU vectors in local coordinates to ENU vectors
            in geocentric coordinates. Shape is (N, 2, 2). To get the opposite rotation,
            use the transpose by swapping the last two axes of the array. The rotation
            matrices are (2, 2), and should be applied on (east, north) components. The
            upward component is the same in the two coordinate systems.
            N is the size of lon and lat (they will be flattened)
        """
        xp = get_array_module(lon, lat)
        lon, lat = (value.reshape(-1) for value in xp.broadcast_arrays(lon, lat))
        geographic_lon, geographic_lat = self.local_to_geographic(lon, lat)
        local_enu_basis = xp.broadcast_to(xp.eye(3), (lon.size, 3, 3))
        local_ecef_basis = enu_to_ecef(local_enu_basis, lat[:, None], lon[:, None])
        local_to_geographic = xp.asarray(self.local_to_geographic_matrix)
        geographic_ecef_basis = xp.einsum("ij,nkj->nki", local_to_geographic, local_ecef_basis)
        geographic_enu_basis = ecef_to_enu(
            geographic_ecef_basis,
            geographic_lat[:, None],
            geographic_lon[:, None],
        )
        return xp.swapaxes(geographic_enu_basis, -1, -2)[:, :2, :2]

    def geographic_vector_to_cube(self, east, north, lon, lat):
        """Project geographic tangent vectors into cube-coordinate components.

        Parameters
        ----------
        east : array-like
            Array of N eastward (geo) components
        north : array-like
            Array of N northward (geo) components
        lon : array-like
            Array of N longitudes that represent vector positions
        lat : array-like
            Array of N latitudes that represent vector positions

        Returns
        -------
        xi : array-like
            N element array of xi coordinates
        eta : array-like
            N element array of eta coordinates
        xi_component : array-like
            N element array of vector components in xi direction
        eta_component : array-like
            N element array of vector components in eta direction

        """
        xp = get_array_module(east, north, lon, lat)
        east, north, lon, lat = (
            value.reshape(-1) for value in xp.broadcast_arrays(east, north, lon, lat)
        )
        xi, eta = self.geographic_to_cube(lon, lat)
        geographic_ecef = enu_to_ecef(
            xp.stack((east, north, xp.zeros_like(east)), axis=1),
            lat,
            lon,
        )
        local_ecef = xp.einsum(
            "ij,nj->ni",
            xp.asarray(self.geographic_to_local_matrix),
            geographic_ecef,
        )
        cube_matrix = cs_vectors._cartesian_to_cube_matrix(
            xi,
            eta,
            radius=1.0,
            face=_NORTH_FACE,
        )
        cube = xp.einsum("nij,nj->ni", cube_matrix, local_ecef)
        return xi, eta, cube[:, 0], cube[:, 1]

    def cube_vector_to_geographic(self, xi_component, eta_component, xi, eta):
        """Convert cube-coordinate tangent components to geographic ENU.

        Parameters
        ----------
        xi_component : array-like
            Array of N xi components
        eta_component : array-like
            Array of N eta components
        xi : array-like
            Array of N xi coords that represent vector positions
        eta : array-like
            Array of N eta coords that represent vector positions

        Returns
        -------
        lon : array-like
            N element array of lon coordinates
        lat : array-like
            N element array of lat coordinates
        east : array-like
            N element array of vector components in east direction
        north : array-like
            N element array of vector components in north direction

        """
        xp = get_array_module(xi_component, eta_component, xi, eta)
        xi_component, eta_component, xi, eta = (
            value.reshape(-1)
            for value in xp.broadcast_arrays(xi_component, eta_component, xi, eta)
        )
        lon, lat = self.cube_to_geographic(xi, eta)
        cube = xp.stack((xi_component, eta_component, xp.zeros_like(xi_component)), axis=1)
        cartesian_matrix = cs_vectors._cube_to_cartesian_matrix(
            xi,
            eta,
            radius=1.0,
            face=_NORTH_FACE,
        )
        local_ecef = xp.einsum("nij,nj->ni", cartesian_matrix, cube)
        geographic_ecef = xp.einsum(
            "ij,nj->ni",
            xp.asarray(self.local_to_geographic_matrix),
            local_ecef,
        )
        geographic = ecef_to_enu(geographic_ecef, lat, lon)
        return lon, lat, geographic[:, 0], geographic[:, 1]

    def differential_elements(self, xi, eta, dxi, deta, radius=1.0):
        """Calculate magnitudes of line and surface elements.

        Implementation of equations 18-20 of Ronchi et al.

        Inputs are broadcast together and all outputs have the resulting
        shape.

        xi, eta, dxi, deta must all be given in radians. dlxi and dleta
        will be given in units of R, and dS in units of R squared (default
        is radian and steradian)

        Parameters
        ----------
        xi : array-like
            xi coordinate(s) of surface element(s)
        eta : array-like
            eta coordinate(s) of surface element(s)
        dxi : array-like
            dimension(s) of surface element(s) in xi direction
        deta : array-like
            dimension(s) of surface element(s) in eta direction
        radius : float, optional
            radius of the sphere - default is 1

        Returns
        -------
        dlxi : array-like
            Length of line element(s), in radians or in units of ``radius``,
            along xi direction
        dleta : array-like
            Length of line element(s), in radians or in units of ``radius``,
            along eta direction
        dS : array-like
            Area(s) of surface element(s), in steradians or in
            squared units of ``radius``
        """
        xp = get_array_module(xi, eta, dxi, deta, radius)
        xi, eta, dxi, deta, radius = xp.broadcast_arrays(xi, eta, dxi, deta, radius)
        metric = cs_coordinates.surface_metric_tensor(xi, eta, radius=radius).reshape(
            xi.shape + (2, 2)
        )

        dlxi = xp.sqrt(metric[..., 0, 0]) * dxi
        dleta = xp.sqrt(metric[..., 1, 1]) * deta
        area_scale = xp.sqrt(metric[..., 0, 0] * metric[..., 1, 1] - metric[..., 0, 1] ** 2)
        dS = area_scale * dxi * deta

        return dlxi, dleta, dS


__all__ = ["RegionalCSProjection"]
