"""Code for working with cubed sphere projection in in a limited region.
A cubed sphere grid is a grid that is defined via the projection of a circumscribed
cube onto a sphere. The great advantage of this grid is that it avoids any pole
problem, and that there is not a large variation in spatial resolution across the
grid. The disadvantage is that it is non-orthogonal, which means that differential
operators change. The purpose of this script is to take care of that problem.

This code only implements a grid on (part of) one side of the cube. The purpose
is to use it for regional data analyses such as SECS, and potentially simple
modelling. The code uses the equations for the north pole side of the cube

The grid and associated math is completely based on:
C. Ronchi, R. Iacono, P.S. Paolucci, The “Cubed Sphere”: A New Method for the
Solution of Partial Differential Equations in Spherical Geometry, Journal of
Computational Physics, Volume 124, Issue 1, 1996, Pages 93-114,
https://doi.org/10.1006/jcph.1996.0047.

KML, May 2020
Updates:
- June 2021: Made differentiation matrix sparse + arbitrary stencil
- October 2021: Fixed issue with xi and eta not going in expected direction
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import sparse

from kompe.core import SphericalRepresentation
from kompe.cubed_sphere import cs_coordinates
from kompe.math import content_fingerprint
from kompe.mesh import StructuredSurfaceMesh

from . import cs_vectors, diffutils, spherical

d2r = np.pi / 180

datapath = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data") + os.sep
)

REGIONAL_CS_GRID_SCHEMA = "kompe.regional_cs_grid"
REGIONAL_CS_GRID_SCHEMA_VERSION = 1


class RegionalCSProjection:
    def __init__(self, position, orientation):
        """Set up cubed sphere projection

        The RegionalCSProjection is set up by
        1) rotating to a local coordinate system in which 'position'
        is at the pole, and 'orientation' defines the x axis (prime meridian)
        2) applying the Ronchi et al. conversions to xi, eta coords on the
        local coordinates

        Parameters
        ----------
        position: array (lon, lat)
            coordinate at which the cube surface should be
            tangential to the sphere - the center of the projection.
            Pair of values for longitude and latitude [deg]
        orientation: scalar or 2-element array-like
            orientation of the cube surface.
            if scalar: angle in degrees, that defines the the xi axis: orientation = 0 / 180
            implies a xi axis in the east-west direction, positive towards east / west.
            orientation = 90 / 270 impliex a xi axis towards north / south.
            if 2-element array-like: The elements denote the eastward and northward components
            of a vector that is aligned with the xi axis.
        """
        self.position = np.array(position)
        self.orientation = np.array(orientation)

        if self.orientation.size == 2:  # interpreted as a east, north component:
            orientation_norm = np.linalg.norm(self.orientation)
            if orientation_norm == 0:
                raise ValueError("orientation must be non-zero")
            if not np.isclose(orientation_norm, 1.0, rtol=0.0, atol=1e-15):
                self.orientation = self.orientation / orientation_norm
        else:  # interpreted as scalar
            assert self.orientation.size == 1, (
                "orientation must be either scalar or have 2 elements"
            )
            self.orientation = np.array([np.cos(orientation * d2r), np.sin(orientation * d2r)])
        v = np.array([self.orientation[0], self.orientation[1], 0]).reshape((1, 3))

        self.lon0, self.lat0 = position

        # the z axis of local coordinat system described in geocentric coords:
        self.z = np.array(
            [
                np.cos(self.lat0 * d2r) * np.cos(self.lon0 * d2r),
                np.cos(self.lat0 * d2r) * np.sin(self.lon0 * d2r),
                np.sin(self.lat0 * d2r),
            ]
        )

        # the x axis is the orientation described in ECEF coords:
        self.y = spherical.enu_to_ecef(v, np.array(self.lon0), np.array(self.lat0)).flatten()

        # the y axis completes the system:
        self.x = np.cross(self.y, self.z)

        # define rotation matrices for rotations between local and geocentric:
        self.R_geo2local = np.vstack(
            (self.x, self.y, self.z)
        )  # rotation matrix from GEO to rotated coords (ECEF)
        self.R_local2geo = self.R_geo2local.T  # inverse

    @property
    def signature(self):
        """Return immutable projection identity for grids and caches."""
        return (
            "REGIONAL_CS_PROJECTION",
            tuple(float(value) for value in self.position),
            tuple(float(value) for value in self.orientation),
        )

    def geo2cube(self, lon, lat, set_points_off_cube_to_nan=False):
        """Convert from geocentric coordinates to cube coords (xi, eta)

        Input parameters must have same shape. Output will have same shape.
        Points that are outside the cube surface will be nans

        Parameters
        ----------
        lon: array
            geocentric longitude(s) [deg] to convert to cube coords
        lat: array:
            geocentric latitude(s) [deg] to convert to cube coords.
        set_points_off_cube_to_nan : bool (optional)
            set to True if points that are not on the cube should be
            set to nan (default is False).

        Returns
        -------
        xi: array
            xi, as defined in Ronchi et al, after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]
        eta: array
            eta, as defined in Ronchi et al., after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]

        """
        lon, lat = np.broadcast_arrays(np.asarray(lon), np.asarray(lat))
        local_lon, local_lat = self.geo2local(lon, lat)
        xi, eta, _ = cs_coordinates.geo_to_cube(local_lon, local_lat, block=4)

        invalid = local_lat < 0
        if set_points_off_cube_to_nan:
            invalid = invalid | (np.deg2rad(90 - local_lat) > np.pi / 4)
        return np.where(invalid, np.nan, xi), np.where(invalid, np.nan, eta)

    def cube2geo(self, xi, eta):
        """Convert from cube coordinates (xi, eta) to geocentric (lon, lat)

        Input parameters must have same shape. Output will have same shape.
        Points that are outside the cube surface will be nans

        Parameters
        ----------
        lon: array
            geocentric longitude(s) [deg] to convert to cube coords
        lat: array:
            geocentric latitude(s) [deg] to convert to cube coords.

        Returns
        -------
        xi: array
            xi, as defined in Ronchi et al., after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]
        eta: array
            eta, as defined in Ronchi et al., after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]


        """
        xi, eta = np.broadcast_arrays(np.asarray(xi, dtype=float), np.asarray(eta, dtype=float))
        _, theta, phi = cs_coordinates.cube_to_spherical(xi, eta, block=4, deg=True)
        return self.local2geo(phi, 90 - theta)

    def geo2local(self, lon, lat, reverse=False):
        """Convert from geocentric coordinates to local coordinates

        lon and lat must have the same shape. Shapes are preserved in output.

        Parameters
        ----------
        lon: array-like
            array of longitudes [deg]
        lat: array-like
            array of latitudes [deg]
        reverse: bool, optional
            set to False (default) if you want to rate from geocentric to local,
            set to True if you want the opposite rotation

        Returns
        -------
        lon: array-like
            array of longitudes [deg] in new coordinate system
        lat: array-like
            array of latitudes [deg] in new coordinate system
        """
        assert lat.shape == lon.shape
        shape = lat.shape

        # set up ECEF position vectors, and rotate using rotation matrices
        lat, lon = np.array(lat).flatten() * d2r, np.array(lon).flatten() * d2r
        r = np.vstack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
        if reverse:
            r_ = self.R_local2geo.dot(r)
        else:
            r_ = self.R_geo2local.dot(r)

        # calcualte spherical coords:
        newlat = np.arcsin(r_[2]) / d2r
        newlon = np.arctan2(r_[1], r_[0]) / d2r

        return (newlon.reshape(shape), newlat.reshape(shape))

    def local2geo(self, lon, lat, reverse=False):
        """Convert from local coordinates to geocentric coordinates

        lon and lat must have the same shape. Shapes are preserved in output

        Parameters
        ----------
        lon: array-like
            array of longitudes [deg]
        lat: array-like
            array of latitudes [deg]
        reverse: bool, optional
            set to False (default) if you want to rate from local to geocentric,
            set to True if you want the opposite rotation

        Returns
        -------
        lon: array-like
            array of longitudes [deg] in new coordinate system
        lat: array-like
            array of latitudes [deg] in new coordinate system

        Note
        ----
        See self.geo2local for implementation
        """
        if reverse:
            return self.geo2local(lon, lat)
        else:
            return self.geo2local(lon, lat, reverse=True)

    def local2geo_enu_rotation(self, lon, lat):
        """Calculate rotation matrices that transform local ENU to geocentric ENU

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
        th = (90 - np.array(lat).flatten()) * d2r
        ph = np.array(lon).flatten() * d2r

        # from ENU to ECEF:
        e_R = np.vstack((-np.sin(ph), np.cos(ph), np.zeros_like(ph))).T  # (N, 3)
        n_R = np.vstack(
            (-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th))
        ).T  # (N, 3)
        u_R = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).T  # (N, 3)

        R_enulocal2eceflocal = np.stack((e_R, n_R, u_R), axis=2)  # (N, 3, 3) with e n u in columns

        # from local to geocentric:
        lon_G, lat_G = self.local2geo(lon, lat)
        th = (90 - lat_G) * d2r
        ph = lon_G * d2r

        e_G = np.vstack((-np.sin(ph), np.cos(ph), np.zeros_like(ph))).T  # (N, 3)
        n_G = np.vstack(
            (-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th))
        ).T  # (N, 3)
        u_G = np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))).T  # (N, 3)

        R_ecefgeo2enugeo = np.stack((e_G, n_G, u_G), axis=1)  # (N, 3, 3) with e n u in rows

        # Combine:
        R_enulocal2ecefgeo = np.einsum("ij , njk -> nik", self.R_local2geo, R_enulocal2eceflocal)
        R_enulocal2enugeo = np.einsum("nij, njk -> nik", R_ecefgeo2enugeo, R_enulocal2ecefgeo)

        # the result should describe a 2D rotation matrix:
        assert np.all(np.isclose(R_enulocal2enugeo[:, 2, 2], 1, atol=1e-6))
        assert np.all(np.isclose(R_enulocal2enugeo[:, 2, np.array([0, 1])], 0, atol=1e-6))
        assert np.all(np.isclose(R_enulocal2enugeo[:, np.array([0, 1]), 2], 0, atol=1e-6))
        return R_enulocal2enugeo[:, :2, :2]  # (N, 2, 2)

    def vector_cube_projection(self, east, north, lon, lat, return_xi_eta=True):
        """Calculate vector components projected on cube

        Perfor vector rotation from geographic system to cube
        system, using self.local2geo_enu_rotation and equation
        (14) of Ronchi et al.

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
        return_xi_eta: bool, optional
            set to False to return only the vector components. If True
            (default), returning the xi, eta coordinates corresponding
            to (lon, lat) as well.

        Returns
        -------
        xi: array-like  (if return_xi_eta is True)
            N element array of xi coordinates
        eta: array-like (if return_xi_eta is True)
            N element array of eta coordinates
        Axi: array-like
            N element array of vector components in xi direction
        Aeta: array-like
            N element array of vector components in eta direction

        """
        east, north, lon, lat = [np.array(x).flatten() for x in [east, north, lon, lat]]
        Ageo = np.vstack((east, north)).T

        # rotation from geo to local:
        local_lon, local_lat = self.geo2local(lon, lat)
        R_enu_global2local = self.local2geo_enu_rotation(local_lon, local_lat)
        Alocal = np.einsum("nji, nj->ni", R_enu_global2local, Ageo).T

        # rearrange to south, east instead of east, north:
        Alocal = np.vstack((-Alocal[1], Alocal[0])).T

        # calculate the parameters used in transformation matrix:
        xi, eta = self.geo2cube(lon, lat)
        X = np.tan(-xi)
        Y = np.tan(-eta)
        delta = 1 + X**2 + Y**2
        C = np.sqrt(1 + X**2)
        D = np.sqrt(1 + Y**2)
        dd = np.sqrt(delta - 1)

        # calculate transformation matrix elements:
        R = np.empty((east.size, 2, 2))
        R[:, 0, 0] = -D * X / dd
        R[:, 0, 1] = D * Y / dd / np.sqrt(delta)
        R[:, 1, 0] = -C * Y / dd
        R[:, 1, 1] = -C * X / dd / np.sqrt(delta)

        # rotate and return
        Acube = np.einsum("nij, nj->ni", R, Alocal).T

        # components in xi and eta directions:
        Axi, Aeta = Acube[0], Acube[1]
        if return_xi_eta:
            return xi, eta, Axi, Aeta
        else:
            return Axi, Aeta

    def vector_cube_to_geo(self, Axi, Aeta, xi, eta, return_lon_lat=True):
        """Calculate vector components projected on cube

        Perfor vector rotation from cube system to geographic
        system, using self.local2geo_enu_rotation and equation
        (14) of Ronchi et al.

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
        return_lon_lat: bool, optional
            set to False to return only the vector components. If True
            (default), returning the lon, lat coordinates corresponding
            to (xi, eta) as well.

        Returns
        -------
        lon: array-like (if return_lon_lat is True)
            N element array of lon coordinates
        lat: array-like (if return_lon_lat is True)
            N element array of lat coordinates
        east: array-like
            N element array of vector components in east direction
        north: array-like
            N element array of vector components in north direction

        """
        Axi, Aeta, xi, eta = [np.array(x).flatten() for x in [Axi, Aeta, xi, eta]]
        Acube = np.vstack((Axi, Aeta)).T

        # calculate the parameters used in transformation matrix:
        X = np.tan(-xi)
        Y = np.tan(-eta)
        delta = 1 + X**2 + Y**2
        C = np.sqrt(1 + X**2)
        D = np.sqrt(1 + Y**2)
        dd = np.sqrt(delta - 1)

        # calculate transformation matrix elements:
        R = np.empty((Axi.size, 2, 2))
        R[:, 0, 0] = -D * X / dd
        R[:, 0, 1] = D * Y / dd / np.sqrt(delta)
        R[:, 1, 0] = -C * Y / dd
        R[:, 1, 1] = -C * X / dd / np.sqrt(delta)

        # rotate and return
        Alocal = np.einsum("nji, nj->ni", R, Acube).T

        # rearrange to east, north instead of south, east:
        Alocal = np.vstack((Alocal[1], -Alocal[0])).T

        # rotation from local to geo:
        lon, lat = self.cube2geo(xi, eta)
        local_lon, local_lat = self.geo2local(lon, lat)
        R_enu_global2local = self.local2geo_enu_rotation(local_lon, local_lat)
        Ageo = np.einsum("nij, nj->ni", R_enu_global2local, Alocal).T

        # components in east, north directions:
        east, north = Ageo[0], Ageo[1]
        if return_lon_lat:
            return lon, lat, east, north
        else:
            return east, north

    def get_projected_coastlines(self, resolution="50m"):
        """Generate coastlines in projected coordinates"""
        coastlines = np.load(datapath + "coastlines_" + resolution + ".npz")
        for key in coastlines:
            lat, lon = coastlines[key]
            yield self.geo2cube(lon, lat)

    def differentials(self, xi, eta, dxi, deta, radius=1.0):
        """Calculate magnitudes of line and surface elements

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
        X = np.tan(xi)
        Y = np.tan(eta)
        delta = 1 + X**2 + Y**2
        C = np.sqrt(1 + X**2)
        D = np.sqrt(1 + Y**2)

        dlxi = radius * D * dxi / (delta * np.cos(xi) ** 2)
        dleta = radius * C * deta / (delta * np.cos(eta) ** 2)

        dS = radius**2 * deta * dxi / (delta ** (3.0 / 2) * np.cos(xi) ** 2 * np.cos(eta) ** 2)

        return dlxi, dleta, dS


class RegionalCSGrid(SphericalRepresentation, StructuredSurfaceMesh):
    def __init__(
        self,
        projection,
        length,
        width,
        length_resolution,
        width_resolution,
        *,
        radius,
        edges=None,
        width_shift=0.0,
    ):
        """Set up a regional cubed-sphere grid.

        Create a regular grid in xi,eta-coordinates. The grid will cover a
        region of the cube surface that is L by W, where L is the dimension along
        the projection.orientation vector. The center of the grid is located at
        projection.position.

        Parameters
        ----------
        projection: RegionalCSProjection
            RegionalCSProjection
        length: float
            Dimension of grid along RegionalCSProjection.orientation, i.e. the "length"
            of the grid. Dimension corresponds to the dimension of R at the
            cube-sphere intersection point
        width: float
            Dimension of grid perpendicular RegionalCSProjection.orientation, i.e. the
            "width" of the grid. Dimension corresponds to the dimension of R at
            the cube-sphere intersection point
        length_resolution: float or int
            Cell size (float) or number of cells (int) in the length direction.
        width_resolution: float or int
            Cell size (float) or number of cells (int) in the width direction.
        width_shift: float, optional
            Distance, in the same units as ``radius``, by which to move the grid
            in the xi-direction,
            or W direction. Positive numbers will move the center right (towards
            positive xi)
        edges: tuple, optional
            if you want to force the grid in xi/eta space to certain values, provide
            them in this tuple.
        radius: float
            Radius of the sphere, in the same units as the dimensions and any
            floating-point resolutions. It is required so that the grid never
            silently assumes kilometres or metres.

        """
        if not isinstance(projection, RegionalCSProjection):
            raise TypeError("projection must be a RegionalCSProjection")
        for name, value in (("length", length), ("width", width), ("radius", radius)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        length_resolution = RegionalCSGridSpec._resolution("length_resolution", length_resolution)
        width_resolution = RegionalCSGridSpec._resolution("width_resolution", width_resolution)
        if isinstance(length_resolution, int) != isinstance(width_resolution, int):
            raise ValueError("length and width resolutions must use the same convention")
        if not np.isfinite(width_shift):
            raise ValueError("width_shift must be finite")

        self.projection = projection
        self.radius = float(radius)
        self.width_shift = float(width_shift)
        self.length = float(length)
        self.width = float(width)
        self.length_resolution = length_resolution
        self.width_resolution = width_resolution
        self.edges = edges

        # make xi and eta arrays for the grid cell boundaries:
        if edges is None:
            if isinstance(length_resolution, int):
                xi_edge = (
                    np.linspace(
                        -np.arctan(length / radius) / 2,
                        np.arctan(length / radius) / 2,
                        width_resolution + 1,
                    )
                    - width_shift / self.radius
                )
                eta_edge = np.linspace(
                    -np.arctan(width / radius) / 2,
                    np.arctan(width / radius) / 2,
                    length_resolution + 1,
                )
            else:
                xi_edge = (
                    np.r_[
                        -np.arctan(length / radius) / 2 : np.arctan(length / radius)
                        / 2 : np.arctan(length_resolution / radius)
                    ]
                    - width_shift / self.radius
                )
                eta_edge = np.r_[
                    -np.arctan(width / radius) / 2 : np.arctan(width / radius) / 2 : np.arctan(
                        width_resolution / radius
                    )
                ]
        else:
            xi_edge, eta_edge = edges

        # outer grid limits in xi and eta coords:
        self.xi_min, self.xi_max = xi_edge.min(), xi_edge.max()
        self.eta_min, self.eta_max = eta_edge.min(), eta_edge.max()

        # number of grid cells in L (eta) and W (xi) directions:
        self.NL, self.NW = len(eta_edge) - 1, len(xi_edge) - 1

        # size of grid cells in xi, eta coordinates:
        self.dxi = xi_edge[1] - xi_edge[0]
        self.deta = eta_edge[1] - eta_edge[0]

        # xi, eta coordinates of cell corners:
        self.xi_mesh, self.eta_mesh = np.meshgrid(xi_edge, eta_edge, indexing="xy")

        # lon, lat coordiantes of cell corners:
        self.lon_mesh, self.lat_mesh = self.projection.cube2geo(self.xi_mesh, self.eta_mesh)

        # xi, eta coordinates of grid points (cell centers):
        self.xi = self.xi_mesh[0:-1, 0:-1] + self.dxi / 2
        self.eta = self.eta_mesh[0:-1, 0:-1] + self.deta / 2

        # geocentric lon, lat [deg] of grid points:
        self.lon, self.lat = self.projection.cube2geo(self.xi, self.eta)
        self.local_lon, self.local_lat = self.projection.geo2local(self.lon, self.lat)

        # Canonical angular coordinates are degrees, matching Grid and every
        # basis evaluator. Explicit radian views keep unit conversions visible.
        self.phi = self.lon.reshape(-1).copy()
        self.theta = (90 - self.lat).reshape(-1)
        self.phi_rad, self.theta_rad = self.phi * d2r, self.theta * d2r

        # cubed square parameters for grid points (cell centers)
        self.X = np.tan(self.xi)
        self.Y = np.tan(self.eta)
        self.delta = 1 + self.X**2 + self.Y**2
        self.C = np.sqrt(1 + self.X**2)
        self.D = np.sqrt(1 + self.Y**2)

        # set size and shape
        self.size = self.lat.size
        self.shape = self.lat.shape

        # calcualte cell area
        self.A = self.projection.differentials(
            self.xi, self.eta, self.dxi, self.deta, radius=self.radius
        )[2]
        self.area_weights = self.A.reshape(-1)

        self._signature = (
            "REGIONAL_CS_GRID",
            self.projection.signature,
            self.radius,
            content_fingerprint(
                {
                    "xi_edges": np.asarray(self.xi_mesh[0], dtype="<f8"),
                    "eta_edges": np.asarray(self.eta_mesh[:, 0], dtype="<f8"),
                }
            ),
        )
        self.validate_metadata()
        self.validate_mesh_metadata()

    @property
    def mesh_shape(self):
        """Logical ``(eta, xi)`` cell shape."""
        return tuple(int(length) for length in self.shape)

    @property
    def cell_center_theta(self):
        """Cell-centre colatitudes in degrees."""
        return 90.0 - self.lat

    @property
    def cell_center_phi(self):
        """Cell-centre longitudes in degrees."""
        return self.lon

    @property
    def cell_areas(self):
        """Cell areas in squared radius units."""
        return self.A.reshape(self.mesh_shape)

    @property
    def kind(self):
        """Short identifier for a regional cubed-sphere grid."""
        return "REGIONAL_CS_GRID"

    @property
    def index_names(self):
        """Names of the structured cell-center indices."""
        return ("eta", "xi")

    @property
    def index_length(self):
        """Number of cell-centered values."""
        return self.size

    @property
    def index_arrays(self):
        """Flattened geographic coordinates for cell-centered values."""
        return self.lat.reshape(-1), self.lon.reshape(-1)

    @property
    def signature(self):
        """Return exact geometry identity."""
        return self._signature

    @property
    def coefficient_space_signature(self):
        """Return grid-value compatibility identity."""
        return self.signature

    @cached_property
    def operators(self):
        """Differential and interpolation operators bound to this grid."""
        return RegionalCSOperators(self)

    def to_spec(self):
        """Return a versioned, JSON-serializable grid specification."""
        return RegionalCSGridSpec.from_grid(self)

    @classmethod
    def from_spec(cls, spec):
        """Construct a grid from a :class:`RegionalCSGridSpec` or mapping."""
        if isinstance(spec, Mapping):
            spec = RegionalCSGridSpec.from_mapping(spec)
        if not isinstance(spec, RegionalCSGridSpec):
            raise TypeError("spec must be a RegionalCSGridSpec or mapping")
        return spec.build()

    def __repr__(self):
        """String representation"""
        th0, th1 = 2 * self.xi.max() / d2r, 2 * self.eta.max() / d2r
        orientation = self.projection.orientation.flatten()[:2]  # east, north components
        lon, lat = (
            self.projection.lon0,
            self.projection.lat0,
        )

        return (
            f"{self.shape[0]} x {self.shape[1]} cubed sphere grid\n"
            + f"Centered at lon, lat = {lon:.1f}, {lat:.1f}\n"
            + f"Orientation: {orientation[0]:.2f} east, {orientation[1]:.2f} north, \n"
            + f"Extent: ~{th0:.1f} x {th1:.1f} degrees central angle"
        )

    def _index(self, i, j):
        """
        Calculate the 1D index that corresponds to the grid index i, j

        Parameters
        ----------
        i: array-like (int)
            row index(es)
        j: array-like (int)
            columns index(es)

        Returns
        -------
        1D array of ints which denote the index(es) of i, j in a flattened version
        of a 2D array of shape (self.NL, self.NW)
        """
        i = np.array(i) % self.NL  # wrap negative indices to other end
        j = np.array(j) % self.NW

        try:
            return np.ravel_multi_index((i, j), (self.NL, self.NW)).flatten()
        except Exception:
            print("invalid index?", i, j, self.NL, self.NW)

    def _index2d(self, index1d):
        """
        Calculate 2d indices from the input 1D index.
        Inverse of _index() function.

        Added 2021-11-02 by JPR

        Parammeters
        -----------
        index1d: array-like (int) of length N of 1d indices to be represented
            by the 2D ij indices

        Returns
        -------
        Two 1D arrays, first containing the i indices, second the j indices
        Same length (N) as input parameter.

        """
        i = index1d // self.shape[1]
        j = index1d % self.shape[1]

        return i, j

    def count(self, lon, lat, **kwargs):
        """
        Count number of points in each grid cell

        Parameters
        ----------
        lon : array
            array of longitudes [degrees]. Must have same size as lat
        lat : array
            array of latitudes [degrees]. Must have same size as lon
        kwargs : dict, optional
            passed to numpy.histogram2d. Use this if you want density,
            normed, or weighted histograms for example.


        Returns
        -------
        count : array
            array with count of how many of the coordinates defined
            by lon, lat are in each grid cell. Same shape as self.lat
            and self.lon
        """
        lon, lat = lon.flatten(), lat.flatten()
        xi, eta = self.projection.geo2cube(lon, lat)

        xi_edges, eta_edges = self.xi_mesh[0, :], self.eta_mesh[:, 0]
        count, xi_, eta_ = np.histogram2d(xi, eta, (xi_edges, eta_edges), **kwargs)

        return count.T  # transpose because xi should be horizontal and eta vertical

    def bin_index(self, lon, lat):
        """
        Find the bin index (i, j) for each pair (lon, lat)

        Parameters
        ----------
        lon : array
            array of longitudes [degrees]. Must have same size as lat
        lat : array
            array of latitudes [degrees]. Must have same size as lon

        Returns
        -------
        i : array
            index array for each point (lon, lat) along axis 0 (eta direction)
            N-dimensional array where N is equal to lon.size and lat.size
        j : array
            index array for each point (lon, lat) along axis 1 (xi direction)
            N-dimensional array where N is equal to lon.size and lat.size


        Note
        ----
        Points that are outside the grid will be given index -1
        """
        lon, lat = lon.flatten(), lat.flatten()
        xi, eta = self.projection.geo2cube(lon, lat, set_points_off_cube_to_nan=False)

        xi_edges, eta_edges = self.xi_mesh[0, :], self.eta_mesh[:, 0]

        i = np.digitize(eta, eta_edges) - 1
        j = np.digitize(xi, xi_edges) - 1

        iii = ~self.ingrid(lon, lat)  # points not in grid
        i[iii] = -1
        j[iii] = -1

        return (i, j)

    def ingrid(self, lon, lat, ext_factor=1.0):
        """
        Determine if lon, lat are inside grid boundaries or not.

        Parameters
        ----------
        lon: array
            array of longitudes [degrees] - must have same shape as lat
        lat: array
            array of latitudes [degrees] - must have same shape as lon
        ext_factor: float or int, optional
            Set ext_factor to a positive/negative float to extend/contract
            ``self.length`` and ``self.width`` by the given factor to include/exclude
            points that are outside/inside the grid. If provided as
            positive/negative int, it will extend/contract the region as
            multiples of the grid spacing.

        Returns
        -------
        array of bools with shape of lon and lat
        """
        lat, lon = np.array(lat), np.array(lon)
        if lon.shape != lat.shape:
            raise Exception("RegionalCSGrid.ingrid: lon and lat must have same shape")
        shape = lon.shape
        lon, lat = lon.flatten(), lat.flatten()

        xi, eta = self.projection.geo2cube(lon, lat, set_points_off_cube_to_nan=False)
        if isinstance(ext_factor, int):
            ximin, ximax = (
                self.xi_mesh.min() - ext_factor * self.dxi,
                self.xi_mesh.max() + ext_factor * self.dxi,
            )
            etamin, etamax = (
                self.eta_mesh.min() - ext_factor * self.deta,
                self.eta_mesh.max() + ext_factor * self.deta,
            )
        else:
            ximin, ximax = self.xi_mesh.min() * ext_factor, self.xi_mesh.max() * ext_factor
            etamin, etamax = self.eta_mesh.min() * ext_factor, self.eta_mesh.max() * ext_factor

        return ((xi < ximax) & (xi > ximin) & (eta < etamax) & (eta > etamin)).reshape(shape)

    def get_grid_boundaries(self, geocentric=True):
        """
        Get grid boundaries for plotting

        Yields tuples of (lon, lat) arrays that outline
        the grid cell boundaries.

        Example:
        --------
        for c in obj.get_grid_boundaries():
            lon, lat = c
            plot(lon, lat, 'k-', transform = ccrs.Geocentric())
        """
        if geocentric:
            x, y = self.lon_mesh, self.lat_mesh
        else:
            x, y = self.xi_mesh, self.eta_mesh

        for i in range(self.NL + self.NW + 2):
            if i < self.NL + 1:
                yield (x[i, :], y[i, :])
            else:
                i = i - self.NL - 1
                yield (x[:, i], y[:, i])

    def _gradient_matrices(self, S=1, return_dxi_deta=False, return_sparse=False):
        """
        Calculate the matrix that produces the derivative in the
        eastward and northward directions of a scalar field
        defined on self

        set return_dxi_deta to True to return the matrices that
        differentiate in cubed sphere coordinates instead of geo

        Parameters
        ----------
        S: int, optional
            Stencil size. Default is 1, in which case derivatives will be calculated
            with a 3-point stencil. With S = 2, a 5-point stencil will be used. etc.
        return_dxi_deta: bool, optional
            Set to True if you want matrices that differentiate in the xi / eta
            directions instead of east /  north
        return_sparse: bool, optional
            Set to True if you want scipy.sparse matrices instead of dense numpy arrays
        """
        dxi = self.dxi
        det = self.deta
        N = self.NL
        M = self.NW

        D_xi = {"rows": [], "cols": [], "elements": []}
        D_et = {"rows": [], "cols": [], "elements": []}

        # index arrays (0 to N, M)
        i_arr = np.arange(N)
        j_arr = np.arange(M)

        # meshgrid versions:
        ii, jj = np.meshgrid(i_arr, j_arr, indexing="xy")

        # inner grid points:
        points = np.r_[-S : S + 1 : 1]
        coefficients = diffutils.stencil(points, order=1)
        i_dx, j_dx = ii[:, S:-S], jj[:, S:-S]
        i_dy, j_dy = ii.T[:, S:-S], jj.T[:, S:-S]

        for ll in range(len(points)):
            D_et["rows"].append(self._index(i_dx, j_dx))
            D_et["cols"].append(self._index(i_dx + points[ll], j_dx))
            D_et["elements"].append(np.full(i_dx.size, coefficients[ll] / det))

            D_xi["rows"].append(self._index(i_dy, j_dy))
            D_xi["cols"].append(self._index(i_dy, j_dy + points[ll]))
            D_xi["elements"].append(np.full(i_dy.size, coefficients[ll] / dxi))

        # boundaries
        for kk in np.arange(0, S)[::-1]:
            # LEFT
            points = np.r_[-kk : S + 1 : 1]
            coefficients = diffutils.stencil(points, order=1)
            i_dx, j_dx = ii[:, kk], jj[:, kk]
            i_dy, j_dy = ii.T[:, kk], jj.T[:, kk]

            for ll in range(len(points)):
                D_et["rows"].append(self._index(i_dx, j_dx))
                D_et["cols"].append(self._index(i_dx + points[ll], j_dx))
                D_et["elements"].append(np.full(i_dx.size, coefficients[ll] / det))

                D_xi["rows"].append(self._index(i_dy, j_dy))
                D_xi["cols"].append(self._index(i_dy, j_dy + points[ll]))
                D_xi["elements"].append(np.full(i_dy.size, coefficients[ll] / dxi))

            # RIGHT
            points = np.r_[-S : kk + 1 : 1]
            coefficients = diffutils.stencil(points, order=1)
            i_dx, j_dx = ii[:, -(kk + 1)], jj[:, -(kk + 1)]
            i_dy, j_dy = ii.T[:, -(kk + 1)], jj.T[:, -(kk + 1)]

            for ll in range(len(points)):
                D_et["rows"].append(self._index(i_dx, j_dx))
                D_et["cols"].append(self._index(i_dx + points[ll], j_dx))
                D_et["elements"].append(np.full(i_dx.size, coefficients[ll] / det))

                D_xi["rows"].append(self._index(i_dy, j_dy))
                D_xi["cols"].append(self._index(i_dy, j_dy + points[ll]))
                D_xi["elements"].append(np.full(i_dy.size, coefficients[ll] / dxi))

        D_xi = {key: np.hstack(D_xi[key]) for key in D_xi.keys()}
        D_et = {key: np.hstack(D_et[key]) for key in D_et.keys()}

        D_xi = sparse.csc_matrix(
            (D_xi["elements"], (D_xi["rows"], D_xi["cols"])), shape=(N * M, N * M)
        )
        D_et = sparse.csc_matrix(
            (D_et["elements"], (D_et["rows"], D_et["cols"])), shape=(N * M, N * M)
        )

        if return_dxi_deta:
            if return_sparse:
                return D_xi, D_et
            else:
                return np.array(D_xi.todense()), np.array(D_et.todense())

        # A scalar gradient is a covector.  Convert its coordinate partials
        # through the exact dual basis of the embedded cubed-sphere surface.
        # This remains well-defined at the projection centre and avoids the
        # 0/0 terms in the historical Ronchi-equation implementation.
        east_coeff, north_coeff, _ = self._surface_geometry()
        Le = sparse.diags(east_coeff[:, 0]) @ D_xi + sparse.diags(east_coeff[:, 1]) @ D_et
        Ln = sparse.diags(north_coeff[:, 0]) @ D_xi + sparse.diags(north_coeff[:, 1]) @ D_et
        if return_sparse:
            return Le, Ln
        else:
            return np.array(Le.todense()), np.array(Ln.todense())

    def _divergence_matrix(self, S=1, return_sparse=False):
        """
        Calculate the matrix that produces the divergence of a vector field

        The returned 2N x N matrix operates on a 1D array that represents a
        vector field. The array must be of length 2N, where N is the number
        of grid cells. The first N elements are the eastward components and
        the last N are the northward components.

        Note - this code is based on equations (12) and (23) of Ronchi. The
        'matrification' is explained in my regional data analysis document;
        it is not super easy to understand it from the code alone.

        Parameters
        ----------
        S: int, optional
            Stencil size. Default is 1, in which case derivatives will be calculated
            with a 3-point stencil. With S = 2, a 5-point stencil will be used. etc.
        return_sparse: bool, optional
            Set to True if you want scipy.sparse matrices instead of dense numpy arrays
        """
        D_xi, D_eta = self._gradient_matrices(S=S, return_dxi_deta=True, return_sparse=True)
        east_coeff, north_coeff, sqrt_g = self._surface_geometry()

        # In curvilinear coordinates,
        # div(V) = 1/sqrt(g) * partial_a(sqrt(g) V^a).
        # The dual-basis coefficients convert geographic east/north vector
        # components into the contravariant components V^xi and V^eta.
        inv_sqrt_g = sparse.diags(1 / sqrt_g)
        east = inv_sqrt_g @ (
            D_xi @ sparse.diags(sqrt_g * east_coeff[:, 0])
            + D_eta @ sparse.diags(sqrt_g * east_coeff[:, 1])
        )
        north = inv_sqrt_g @ (
            D_xi @ sparse.diags(sqrt_g * north_coeff[:, 0])
            + D_eta @ sparse.diags(sqrt_g * north_coeff[:, 1])
        )
        result = sparse.hstack((east, north), format="csc")
        return result if return_sparse else result.toarray()

    def _surface_geometry(self):
        """Return geographic dual-basis coefficients and ``sqrt(det(g))``.

        The returned east and north arrays have columns for the xi and eta
        contravariant directions.  They serve both scalar-gradient and vector-
        divergence operators, keeping those two constructions geometrically
        consistent.
        """
        xi = self.xi.reshape(-1)
        eta = self.eta.reshape(-1)

        # Columns 0 and 1 of the inverse coordinate transform are the physical
        # tangent vectors d(position)/dxi and d(position)/deta in local ECEF.
        jacobian_local = cs_vectors.pc(xi, eta, r=self.radius, block=4, inverse=True)[:, :, :2]
        jacobian_geo = np.einsum("ij,njk->nik", self.projection.R_local2geo, jacobian_local)
        metric = np.einsum("nki,nkj->nij", jacobian_geo, jacobian_geo)
        dual_basis = np.einsum("nki,nij->nkj", jacobian_geo, np.linalg.inv(metric))

        lon = np.deg2rad(self.lon.reshape(-1))
        lat = np.deg2rad(self.lat.reshape(-1))
        east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)))
        north = np.column_stack(
            (
                -np.sin(lat) * np.cos(lon),
                -np.sin(lat) * np.sin(lon),
                np.cos(lat),
            )
        )
        east_coeff = np.einsum("nk,nkj->nj", east, dual_basis)
        north_coeff = np.einsum("nk,nkj->nj", north, dual_basis)
        sqrt_g = np.sqrt(np.linalg.det(metric))
        return east_coeff, north_coeff, sqrt_g

    def _interpolate_scalar(self, lon, lat, scalar_field):
        """
        Interpolate values of a cell-centred scalar field at the requested
        longitude/latitude locations. Bilinear interpolation uses only the
        nearest four values and therefore requires memory proportional to the
        number of evaluation points, not the product of grid and point counts.

        Parameters
        ----------
        lon: array
            array of longitudes [degrees] - must have same shape as lat_
        lat: array
            array of latitudes [degrees] - must have same shape as lon_
        scalar_field: array
            2D array (or flattened) of the scalar field as defined on the CS grid
            (dimensions must match)

        Returns
        -------
        Interpolated values of the 2D scalar field at the desired input (lon, lat)
        locations.
        """
        lon, lat = np.broadcast_arrays(np.asarray(lon), np.asarray(lat))

        # Remove points outside grid
        inside = self.ingrid(lon, lat)
        lon_ = lon[inside]
        lat_ = lat[inside]

        # Get i,j index of each evaluation location, as a float
        binnumber = self.bin_index(lon_, lat_)
        i = binnumber[0].flatten()
        j = binnumber[1].flatten()
        xi_obs, eta_obs = self.projection.geo2cube(lon_.flatten(), lat_.flatten())
        xi_grid = self.xi[i, j]
        eta_grid = self.eta[i, j]
        i_frac = (eta_obs - eta_grid) / self.deta
        j_frac = (xi_obs - xi_grid) / self.dxi
        i = i + i_frac
        j = j + j_frac

        # Handle issue with machine precission causing points to end up in wrong place
        # using floor/ceil if specifying to evaluate very close to the grid loations
        # used in the interpolation
        close_i = np.isclose(np.round(i), i, atol=1e-14)
        i[close_i] = np.round(i)[close_i] + 1e-14
        close_j = np.isclose(np.round(j), j, atol=1e-14)
        j[close_j] = np.round(j)[close_j] + 1e-14

        # Handle input points falling within the perimiter cells, but outside the
        # boundary determined by the center location in the perimiter cells. For those
        # input points an extrapolation will be performed based on the bilinear
        # interpolation scheme (just evaluated outside the "box" of the 4 points).
        small_i = i <= 0  # due to ingrid it must also be > -0.5
        i[small_i] = 0.00001
        small_j = j <= 0  # due to ingrid it must also be > -0.5
        j[small_j] = 0.00001
        large_i = i >= self.shape[0] - 1  # due to ingrid it must also be > -0.5
        i[large_i] = self.shape[0] - 1 - 0.00001
        large_j = j >= self.shape[1] - 1  # due to ingrid it must also be > -0.5
        j[large_j] = self.shape[1] - 1 - 0.00001

        # Indices of the four nodes surrounding each observation/evaluation location
        ifloor = np.floor(i).astype(int)
        iceil = np.ceil(i).astype(int)
        jfloor = np.floor(j).astype(int)
        jceil = np.ceil(j).astype(int)

        # CS coordinates and 1D indices of these points
        xi1 = self.xi[ifloor, jfloor]
        eta1 = self.eta[ifloor, jfloor]
        ij1 = np.ravel_multi_index((ifloor, jfloor), (self.NL, self.NW)).flatten()
        # xi2 = grid.xi[iceil,jfloor] # We dont need all 4 the xi-eta coords. since their
        # coordinates will be the pairwise similar
        eta2 = self.eta[iceil, jfloor]
        ij2 = np.ravel_multi_index((iceil, jfloor), (self.NL, self.NW)).flatten()
        # xi3 = grid.xi[iceil,jceil]
        # eta3 = grid.eta[iceil,jceil]
        ij3 = np.ravel_multi_index((iceil, jceil), (self.NL, self.NW)).flatten()
        xi4 = self.xi[ifloor, jceil]
        # eta4 = grid.eta[ifloor,jceil]
        ij4 = np.ravel_multi_index((ifloor, jceil), (self.NL, self.NW)).flatten()

        # CS coordinates of observations/evaluation locations
        xi_obs, eta_obs = self.projection.geo2cube(lon_.flatten(), lat_.flatten())
        # Bilinear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
        w1 = (xi4 - xi_obs) * (eta2 - eta_obs) / ((xi4 - xi1) * (eta2 - eta1))  # w11
        w2 = (xi4 - xi_obs) * (eta_obs - eta1) / ((xi4 - xi1) * (eta2 - eta1))  # w12
        w3 = (xi_obs - xi1) * (eta_obs - eta1) / ((xi4 - xi1) * (eta2 - eta1))  # w22
        w4 = (xi_obs - xi1) * (eta2 - eta_obs) / ((xi4 - xi1) * (eta2 - eta1))  # w21

        scalar_values = np.asarray(scalar_field).reshape(-1)
        if scalar_values.size != self.size:
            raise ValueError(
                f"scalar_field must contain {self.size} grid values; got {scalar_values.size}"
            )
        interpolated_ = (
            w1 * scalar_values[ij1]
            + w2 * scalar_values[ij2]
            + w3 * scalar_values[ij3]
            + w4 * scalar_values[ij4]
        )

        # Insert nans and ensure to return an array with same shape as input
        interpolated = np.full(
            lat.size,
            np.nan,
            dtype=np.result_type(interpolated_, float),
        )
        interpolated[inside.flatten()] = interpolated_

        return interpolated.reshape(lat.shape)


class RegionalCSOperators:
    """Numerical operators associated with one regional cubed-sphere grid.

    The operator object is deliberately separate from the grid geometry. It
    keeps discretization choices explicit while sharing the grid's coordinate
    data and exact geometry identity.
    """

    def __init__(self, grid):
        if not isinstance(grid, RegionalCSGrid):
            raise TypeError("grid must be a RegionalCSGrid")
        self.grid = grid

    @property
    def signature(self):
        """Return the identity of the operator family and its grid."""
        return ("REGIONAL_CS_OPERATORS", self.grid.signature)

    def gradient_matrices(
        self,
        stencil_size=1,
        *,
        cube_coordinates=False,
        sparse=True,
    ):
        """Return scalar-gradient matrices for the grid cell centres."""
        return self.grid._gradient_matrices(
            S=stencil_size,
            return_dxi_deta=cube_coordinates,
            return_sparse=sparse,
        )

    def divergence_matrix(self, stencil_size=1, *, sparse=True):
        """Return the matrix mapping horizontal vectors to divergence."""
        return self.grid._divergence_matrix(S=stencil_size, return_sparse=sparse)

    def interpolate_scalar(self, lon, lat, values):
        """Interpolate cell-centred scalar values to longitude/latitude points."""
        return self.grid._interpolate_scalar(lon, lat, values)

    def surface_geometry(self):
        """Return dual-basis coefficients and the surface metric density."""
        return self.grid._surface_geometry()


@dataclass(frozen=True)
class RegionalCSGridSpec:
    """Versioned, consumer-neutral specification for a regional CS grid."""

    position: tuple[float, float]
    orientation: tuple[float, float]
    length: float
    width: float
    length_resolution: int | float
    width_resolution: int | float
    radius: float
    width_shift: float = 0.0
    edges: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    schema: str = REGIONAL_CS_GRID_SCHEMA
    version: int = REGIONAL_CS_GRID_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema != REGIONAL_CS_GRID_SCHEMA:
            raise ValueError(f"Unsupported regional-grid schema: {self.schema!r}")
        if self.version != REGIONAL_CS_GRID_SCHEMA_VERSION:
            raise ValueError(f"Unsupported regional-grid schema version: {self.version!r}")

        position = self._coordinate_pair("position", self.position)
        orientation = self._coordinate_pair("orientation", self.orientation)
        if np.linalg.norm(orientation) == 0:
            raise ValueError("orientation must be non-zero")
        orientation_norm = np.linalg.norm(orientation)
        if not np.isclose(orientation_norm, 1.0, rtol=0.0, atol=1e-15):
            orientation = tuple(
                float(value) for value in np.asarray(orientation) / orientation_norm
            )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation", orientation)

        for name in ("length", "width", "radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, value)

        width_shift = float(self.width_shift)
        if not np.isfinite(width_shift):
            raise ValueError("width_shift must be finite")
        object.__setattr__(self, "width_shift", width_shift)

        resolutions = (
            self._resolution("length_resolution", self.length_resolution),
            self._resolution("width_resolution", self.width_resolution),
        )
        if isinstance(resolutions[0], int) != isinstance(resolutions[1], int):
            raise ValueError("length and width resolutions must use the same convention")
        object.__setattr__(self, "length_resolution", resolutions[0])
        object.__setattr__(self, "width_resolution", resolutions[1])
        object.__setattr__(self, "edges", self._edges(self.edges))

    @staticmethod
    def _coordinate_pair(name, values):
        try:
            result = tuple(float(value) for value in values)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must contain two numbers") from error
        if len(result) != 2 or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain two finite numbers")
        return result

    @staticmethod
    def _resolution(name, value):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer cell count or float cell size")
        if isinstance(value, (int, np.integer)):
            value = int(value)
        elif isinstance(value, (float, np.floating)):
            value = float(value)
        else:
            raise TypeError(f"{name} must be an integer cell count or float cell size")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    @staticmethod
    def _edges(edges):
        if edges is None:
            return None
        if len(edges) != 2:
            raise ValueError("edges must contain xi and eta edge arrays")
        result = tuple(tuple(float(value) for value in axis) for axis in edges)
        for axis in result:
            values = np.asarray(axis)
            if values.size < 2 or not np.isfinite(values).all():
                raise ValueError("each edge axis must contain at least two finite values")
            if not np.all(np.diff(values) > 0):
                raise ValueError("edge coordinates must be strictly increasing")
        return result

    @classmethod
    def from_grid(cls, grid):
        """Create a specification from a canonical grid."""
        if not isinstance(grid, RegionalCSGrid):
            raise TypeError("grid must be a RegionalCSGrid")
        edges = None
        if grid.edges is not None:
            edges = tuple(tuple(float(value) for value in axis) for axis in grid.edges)
        return cls(
            position=tuple(grid.projection.position),
            orientation=tuple(grid.projection.orientation),
            length=grid.length,
            width=grid.width,
            length_resolution=grid.length_resolution,
            width_resolution=grid.width_resolution,
            radius=grid.radius,
            width_shift=grid.width_shift,
            edges=edges,
        )

    @classmethod
    def from_mapping(cls, metadata):
        """Parse the versioned canonical mapping format."""
        if not isinstance(metadata, Mapping):
            raise TypeError("regional-grid metadata must be a mapping")
        projection = metadata.get("projection")
        if not isinstance(projection, Mapping):
            raise ValueError("projection metadata must be a mapping")
        return cls(
            schema=metadata.get("schema"),
            version=metadata.get("version"),
            position=projection.get("position"),
            orientation=projection.get("orientation"),
            length=metadata.get("length"),
            width=metadata.get("width"),
            length_resolution=metadata.get("length_resolution"),
            width_resolution=metadata.get("width_resolution"),
            radius=metadata.get("radius"),
            width_shift=metadata.get("width_shift", 0.0),
            edges=metadata.get("edges"),
        )

    def to_mapping(self):
        """Return the stable JSON-compatible representation."""
        return {
            "schema": self.schema,
            "version": self.version,
            "projection": {
                "position": list(self.position),
                "orientation": list(self.orientation),
            },
            "length": self.length,
            "width": self.width,
            "length_resolution": self.length_resolution,
            "width_resolution": self.width_resolution,
            "radius": self.radius,
            "width_shift": self.width_shift,
            "edges": None if self.edges is None else [list(axis) for axis in self.edges],
        }

    def build(self):
        """Build a regional cubed-sphere grid from this specification."""
        edges = None
        if self.edges is not None:
            edges = tuple(np.asarray(axis) for axis in self.edges)
        projection = RegionalCSProjection(self.position, self.orientation)
        return RegionalCSGrid(
            projection,
            self.length,
            self.width,
            self.length_resolution,
            self.width_resolution,
            radius=self.radius,
            edges=edges,
            width_shift=self.width_shift,
        )


__all__ = [
    "REGIONAL_CS_GRID_SCHEMA",
    "REGIONAL_CS_GRID_SCHEMA_VERSION",
    "RegionalCSGrid",
    "RegionalCSGridSpec",
    "RegionalCSOperators",
    "RegionalCSProjection",
]
