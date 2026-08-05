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
from scipy import sparse as scipy_sparse

from kompe.cubed_sphere import cs_coordinates
from kompe.grid import SphericalGrid
from kompe.math import as_linear_map, content_fingerprint
from kompe.mesh import StructuredSurfaceMesh

from . import cs_vectors, diffutils, spherical

d2r = np.pi / 180

datapath = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data") + os.sep
)

REGIONAL_CS_MESH_SCHEMA = "kompe.regional_cs_mesh"
REGIONAL_CS_MESH_SCHEMA_VERSION = 1


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
            self.orientation = np.array([np.cos(angle * d2r), np.sin(angle * d2r)])
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
        for name in (
            "position",
            "orientation",
            "x",
            "y",
            "z",
            "R_geo2local",
            "R_local2geo",
        ):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            setattr(self, name, values)

    @property
    def signature(self):
        """Return immutable projection identity for grids and caches."""
        return (
            "REGIONAL_CS_PROJECTION",
            tuple(float(value) for value in self.position),
            tuple(float(value) for value in self.orientation),
        )

    def geographic_to_cube(self, lon, lat, set_points_off_cube_to_nan=False):
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
        local_lon, local_lat = self.geographic_to_local(lon, lat)
        xi, eta, _ = cs_coordinates.geo_to_cube(local_lon, local_lat, block=4)

        invalid = local_lat < 0
        if set_points_off_cube_to_nan:
            invalid = invalid | (np.deg2rad(90 - local_lat) > np.pi / 4)
        return np.where(invalid, np.nan, xi), np.where(invalid, np.nan, eta)

    def cube_to_geographic(self, xi, eta):
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
        return self.local_to_geographic(phi, 90 - theta)

    def geographic_to_local(self, lon, lat, reverse=False):
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

    def local_to_geographic(self, lon, lat, reverse=False):
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
        See self.geographic_to_local for implementation
        """
        if reverse:
            return self.geographic_to_local(lon, lat)
        else:
            return self.geographic_to_local(lon, lat, reverse=True)

    def local_to_geographic_enu_rotation(self, lon, lat):
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
        lon_G, lat_G = self.local_to_geographic(lon, lat)
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

    def geographic_vector_to_cube(self, east, north, lon, lat, return_xi_eta=True):
        """Calculate vector components projected on cube

        Perfor vector rotation from geographic system to cube
        system, using self.local_to_geographic_enu_rotation and equation
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
        local_lon, local_lat = self.geographic_to_local(lon, lat)
        R_enu_global2local = self.local_to_geographic_enu_rotation(local_lon, local_lat)
        Alocal = np.einsum("nji, nj->ni", R_enu_global2local, Ageo).T

        # rearrange to south, east instead of east, north:
        Alocal = np.vstack((-Alocal[1], Alocal[0])).T

        # calculate the parameters used in transformation matrix:
        xi, eta = self.geographic_to_cube(lon, lat)
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

    def cube_vector_to_geographic(self, Axi, Aeta, xi, eta, return_lon_lat=True):
        """Calculate vector components projected on cube

        Perfor vector rotation from cube system to geographic
        system, using self.local_to_geographic_enu_rotation and equation
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
        lon, lat = self.cube_to_geographic(xi, eta)
        local_lon, local_lat = self.geographic_to_local(lon, lat)
        R_enu_global2local = self.local_to_geographic_enu_rotation(local_lon, local_lat)
        Ageo = np.einsum("nij, nj->ni", R_enu_global2local, Alocal).T

        # components in east, north directions:
        east, north = Ageo[0], Ageo[1]
        if return_lon_lat:
            return lon, lat, east, north
        else:
            return east, north

    def projected_coastlines(self, resolution="50m"):
        """Generate coastlines in projected coordinates"""
        coastlines = np.load(datapath + "coastlines_" + resolution + ".npz")
        for key in coastlines:
            lat, lon = coastlines[key]
            yield self.geographic_to_cube(lon, lat)

    def differential_elements(self, xi, eta, dxi, deta, radius=1.0):
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


class RegionalCSMesh(StructuredSurfaceMesh):
    def __init__(
        self,
        projection,
        length,
        width,
        *,
        radius,
        shape=None,
        cell_size=None,
        xi_edges=None,
        eta_edges=None,
        xi_shift=0.0,
    ):
        """Construct a regional cubed-sphere mesh.

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
        radius: float
            Radius of the sphere, in the same units as the dimensions and any
            cell sizes. It is required so that the mesh never
            silently assumes kilometres or metres.
        shape: tuple of int, optional
            Number of cells along the ``(eta, xi)`` axes.
        cell_size: tuple of float, optional
            Target physical cell sizes along the ``(eta, xi)`` axes. The final
            uniform spacing is adjusted slightly so the requested extent is exact.
        xi_edges, eta_edges: array-like, optional
            Explicit uniformly spaced computational-coordinate edges in radians.
        xi_shift: float, optional
            Physical displacement along the xi axis, in the same units as ``radius``.

        Notes
        -----
        Provide exactly one construction mode: ``shape``, ``cell_size``, or both
        explicit edge arrays. Keeping cell counts and physical cell sizes in
        separate parameters avoids the historical int-versus-float ambiguity.

        """
        if not isinstance(projection, RegionalCSProjection):
            raise TypeError("projection must be a RegionalCSProjection")
        for name, value in (("length", length), ("width", width), ("radius", radius)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        shape = RegionalCSMeshSpec._shape(shape)
        cell_size = RegionalCSMeshSpec._cell_size(cell_size)
        xi_edges = RegionalCSMeshSpec._edge_axis("xi_edges", xi_edges)
        eta_edges = RegionalCSMeshSpec._edge_axis("eta_edges", eta_edges)
        explicit_edges = xi_edges is not None or eta_edges is not None
        if explicit_edges and (xi_edges is None or eta_edges is None):
            raise ValueError("xi_edges and eta_edges must be provided together")
        mode_count = int(shape is not None) + int(cell_size is not None) + int(explicit_edges)
        if mode_count != 1:
            raise ValueError("Provide exactly one of shape, cell_size, or explicit edges")
        if not np.isfinite(xi_shift):
            raise ValueError("xi_shift must be finite")

        self.projection = projection
        self.radius = float(radius)
        self.xi_shift = float(xi_shift)
        self.length = float(length)
        self.width = float(width)
        self.requested_shape = shape
        self.requested_cell_size = cell_size

        xi_span = np.arctan(self.length / self.radius)
        eta_span = np.arctan(self.width / self.radius)
        if shape is not None:
            n_eta, n_xi = shape
            xi_edge = np.linspace(-xi_span / 2, xi_span / 2, n_xi + 1)
            eta_edge = np.linspace(-eta_span / 2, eta_span / 2, n_eta + 1)
        elif cell_size is not None:
            eta_cell_size, xi_cell_size = cell_size
            n_eta = max(1, int(round(eta_span / np.arctan(eta_cell_size / self.radius))))
            n_xi = max(1, int(round(xi_span / np.arctan(xi_cell_size / self.radius))))
            xi_edge = np.linspace(-xi_span / 2, xi_span / 2, n_xi + 1)
            eta_edge = np.linspace(-eta_span / 2, eta_span / 2, n_eta + 1)
        else:
            xi_edge = np.asarray(xi_edges)
            eta_edge = np.asarray(eta_edges)

        xi_edge = xi_edge - self.xi_shift / self.radius
        self.xi_edges = tuple(float(value) for value in xi_edge)
        self.eta_edges = tuple(float(value) for value in eta_edge)

        for name, axis in (("xi", xi_edge), ("eta", eta_edge)):
            spacing = np.diff(axis)
            if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-15):
                raise ValueError(f"{name} edges must be uniformly spaced")

        # outer grid limits in xi and eta coords:
        self.xi_min, self.xi_max = xi_edge.min(), xi_edge.max()
        self.eta_min, self.eta_max = eta_edge.min(), eta_edge.max()

        # number of grid cells in L (eta) and W (xi) directions:
        self.n_eta, self.n_xi = len(eta_edge) - 1, len(xi_edge) - 1

        # size of grid cells in xi, eta coordinates:
        self.dxi = xi_edge[1] - xi_edge[0]
        self.deta = eta_edge[1] - eta_edge[0]

        # xi, eta coordinates of cell corners:
        self.xi_mesh, self.eta_mesh = np.meshgrid(xi_edge, eta_edge, indexing="xy")

        # lon, lat coordiantes of cell corners:
        self.lon_mesh, self.lat_mesh = self.projection.cube_to_geographic(
            self.xi_mesh, self.eta_mesh
        )

        # xi, eta coordinates of grid points (cell centers):
        self.xi = self.xi_mesh[0:-1, 0:-1] + self.dxi / 2
        self.eta = self.eta_mesh[0:-1, 0:-1] + self.deta / 2

        # geocentric lon, lat [deg] of grid points:
        self.lon, self.lat = self.projection.cube_to_geographic(self.xi, self.eta)
        self.local_lon, self.local_lat = self.projection.geographic_to_local(self.lon, self.lat)

        # set size and shape
        self._shape = tuple(int(length) for length in self.lat.shape)

        # calcualte cell area
        self._cell_areas = self.projection.differential_elements(
            self.xi, self.eta, self.dxi, self.deta, radius=self.radius
        )[2]

        self._signature = (
            "REGIONAL_CS_MESH",
            self.projection.signature,
            self.radius,
            content_fingerprint(
                {
                    "xi_edges": np.asarray(self.xi_mesh[0], dtype="<f8"),
                    "eta_edges": np.asarray(self.eta_mesh[:, 0], dtype="<f8"),
                }
            ),
        )
        for name in (
            "xi_mesh",
            "eta_mesh",
            "lon_mesh",
            "lat_mesh",
            "xi",
            "eta",
            "lon",
            "lat",
            "local_lon",
            "local_lat",
            "_cell_areas",
        ):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            setattr(self, name, values)
        self.validate_mesh_metadata()

    @property
    def shape(self):
        """Logical ``(eta, xi)`` cell shape."""
        return self._shape

    @cached_property
    def cell_centers(self):
        """Cell-centre coordinates and physical area weights."""
        return SphericalGrid(lat=self.lat, lon=self.lon, area_weights=self.cell_areas)

    @property
    def cell_areas(self):
        """Cell areas in squared radius units."""
        return self._cell_areas

    @property
    def signature(self):
        """Return exact geometry identity."""
        return self._signature

    @cached_property
    def operators(self):
        """Differential and interpolation operators bound to this grid."""
        return RegionalCSOperators(self)

    def to_spec(self):
        """Return a versioned, JSON-serializable grid specification."""
        return RegionalCSMeshSpec.from_mesh(self)

    @classmethod
    def from_spec(cls, spec):
        """Construct a grid from a :class:`RegionalCSMeshSpec` or mapping."""
        if isinstance(spec, Mapping):
            spec = RegionalCSMeshSpec.from_dict(spec)
        if not isinstance(spec, RegionalCSMeshSpec):
            raise TypeError("spec must be a RegionalCSMeshSpec or mapping")
        return spec.to_mesh()

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

    def flat_index(self, eta_index, xi_index):
        """Return flattened indices for structured cell indices.

        Parameters
        ----------
        eta_index: array-like of int
            Row indices along the eta axis. Negative indices wrap.
        xi_index: array-like of int
            Column indices along the xi axis. Negative indices wrap.

        Returns
        -------
        1D array of ints which denote the index(es) of i, j in a flattened version
        of a 2D array of shape (self.n_eta, self.n_xi)
        """
        i = np.asarray(eta_index) % self.n_eta
        j = np.asarray(xi_index) % self.n_xi

        return np.ravel_multi_index((i, j), (self.n_eta, self.n_xi)).flatten()

    def unravel_index(self, flat_index):
        """Return eta and xi indices for flattened cell indices.

        Parammeters
        -----------
        flat_index: array-like of int
            Flattened cell indices.

        Returns
        -------
        Two 1D arrays, first containing the i indices, second the j indices
        Same length (N) as input parameter.

        """
        return np.unravel_index(flat_index, self.shape)

    def count_points(self, lon, lat, **kwargs):
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
        xi, eta = self.projection.geographic_to_cube(lon, lat)

        xi_edges, eta_edges = self.xi_mesh[0, :], self.eta_mesh[:, 0]
        count, _, _ = np.histogram2d(xi, eta, (xi_edges, eta_edges), **kwargs)

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
        xi, eta = self.projection.geographic_to_cube(lon, lat, set_points_off_cube_to_nan=False)

        xi_edges, eta_edges = self.xi_mesh[0, :], self.eta_mesh[:, 0]

        i = np.digitize(eta, eta_edges) - 1
        j = np.digitize(xi, xi_edges) - 1

        iii = ~self.contains(lon, lat)  # points not in grid
        i[iii] = -1
        j[iii] = -1

        return (i, j)

    def contains(self, lon, lat, extent_factor=1.0):
        """
        Determine if lon, lat are inside grid boundaries or not.

        Parameters
        ----------
        lon: array
            array of longitudes [degrees] - must have same shape as lat
        lat: array
            array of latitudes [degrees] - must have same shape as lon
        extent_factor: float or int, optional
            Set extent_factor to a positive/negative float to extend/contract
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
            raise ValueError("RegionalCSMesh.contains: lon and lat must have same shape")
        shape = lon.shape
        lon, lat = lon.flatten(), lat.flatten()

        xi, eta = self.projection.geographic_to_cube(lon, lat, set_points_off_cube_to_nan=False)
        if isinstance(extent_factor, int):
            ximin, ximax = (
                self.xi_mesh.min() - extent_factor * self.dxi,
                self.xi_mesh.max() + extent_factor * self.dxi,
            )
            etamin, etamax = (
                self.eta_mesh.min() - extent_factor * self.deta,
                self.eta_mesh.max() + extent_factor * self.deta,
            )
        else:
            ximin, ximax = self.xi_mesh.min() * extent_factor, self.xi_mesh.max() * extent_factor
            etamin, etamax = (
                self.eta_mesh.min() * extent_factor,
                self.eta_mesh.max() * extent_factor,
            )

        return ((xi < ximax) & (xi > ximin) & (eta < etamax) & (eta > etamin)).reshape(shape)

    def geographic_boundaries(self, geocentric=True):
        """
        Get grid boundaries for plotting

        Yields tuples of (lon, lat) arrays that outline
        the grid cell boundaries.

        Example:
        --------
        for c in obj.geographic_boundaries():
            lon, lat = c
            plot(lon, lat, 'k-', transform = ccrs.Geocentric())
        """
        if geocentric:
            x, y = self.lon_mesh, self.lat_mesh
        else:
            x, y = self.xi_mesh, self.eta_mesh

        for i in range(self.n_eta + self.n_xi + 2):
            if i < self.n_eta + 1:
                yield (x[i, :], y[i, :])
            else:
                i = i - self.n_eta - 1
                yield (x[:, i], y[:, i])


class RegionalCSOperators:
    """Numerical operators associated with one regional cubed-sphere grid.

    The grid owns coordinates, cells, and topology. This object owns the
    discretization choices and constructs interpolation and differential
    operators against that immutable geometry.
    """

    def __init__(self, mesh):
        if not isinstance(mesh, RegionalCSMesh):
            raise TypeError("mesh must be a RegionalCSMesh")
        self.mesh = mesh

    @property
    def signature(self):
        """Return the identity of the operator family and its grid."""
        return ("REGIONAL_CS_OPERATORS", self.mesh.signature)

    def surface_gradient_matrices(
        self,
        stencil_size=1,
        *,
        cube_coordinates=False,
        sparse=True,
    ):
        """
        Calculate the matrix that produces the derivative in the
        eastward and northward directions of a scalar field
        defined on self

        set return_dxi_deta to True to return the matrices that
        differentiate in cubed sphere coordinates instead of geo

        Parameters
        ----------
        stencil_size: int, optional
            Stencil size. Default is 1, in which case derivatives will be calculated
            with a 3-point stencil. With S = 2, a 5-point stencil will be used. etc.
        cube_coordinates: bool, optional
            Set to True if you want matrices that differentiate in the xi / eta
            directions instead of east /  north
        sparse: bool, optional
            Set to True if you want scipy.sparse matrices instead of dense numpy arrays
        """
        grid = self.mesh
        if isinstance(stencil_size, bool) or not isinstance(stencil_size, (int, np.integer)):
            raise TypeError("stencil_size must be an integer")
        S = int(stencil_size)
        if S < 1:
            raise ValueError("stencil_size must be positive")
        if min(grid.shape) < 2 * S + 1:
            raise ValueError(
                "stencil_size requires at least 2*stencil_size+1 cells along each axis"
            )
        dxi = grid.dxi
        det = grid.deta
        N = grid.n_eta
        M = grid.n_xi

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
            D_et["rows"].append(grid.flat_index(i_dx, j_dx))
            D_et["cols"].append(grid.flat_index(i_dx + points[ll], j_dx))
            D_et["elements"].append(np.full(i_dx.size, coefficients[ll] / det))

            D_xi["rows"].append(grid.flat_index(i_dy, j_dy))
            D_xi["cols"].append(grid.flat_index(i_dy, j_dy + points[ll]))
            D_xi["elements"].append(np.full(i_dy.size, coefficients[ll] / dxi))

        # boundaries
        for kk in np.arange(0, S)[::-1]:
            # LEFT
            points = np.r_[-kk : S + 1 : 1]
            coefficients = diffutils.stencil(points, order=1)
            i_dx, j_dx = ii[:, kk], jj[:, kk]
            i_dy, j_dy = ii.T[:, kk], jj.T[:, kk]

            for ll in range(len(points)):
                D_et["rows"].append(grid.flat_index(i_dx, j_dx))
                D_et["cols"].append(grid.flat_index(i_dx + points[ll], j_dx))
                D_et["elements"].append(np.full(i_dx.size, coefficients[ll] / det))

                D_xi["rows"].append(grid.flat_index(i_dy, j_dy))
                D_xi["cols"].append(grid.flat_index(i_dy, j_dy + points[ll]))
                D_xi["elements"].append(np.full(i_dy.size, coefficients[ll] / dxi))

            # RIGHT
            points = np.r_[-S : kk + 1 : 1]
            coefficients = diffutils.stencil(points, order=1)
            i_dx, j_dx = ii[:, -(kk + 1)], jj[:, -(kk + 1)]
            i_dy, j_dy = ii.T[:, -(kk + 1)], jj.T[:, -(kk + 1)]

            for ll in range(len(points)):
                D_et["rows"].append(grid.flat_index(i_dx, j_dx))
                D_et["cols"].append(grid.flat_index(i_dx + points[ll], j_dx))
                D_et["elements"].append(np.full(i_dx.size, coefficients[ll] / det))

                D_xi["rows"].append(grid.flat_index(i_dy, j_dy))
                D_xi["cols"].append(grid.flat_index(i_dy, j_dy + points[ll]))
                D_xi["elements"].append(np.full(i_dy.size, coefficients[ll] / dxi))

        D_xi = {key: np.hstack(D_xi[key]) for key in D_xi}
        D_et = {key: np.hstack(D_et[key]) for key in D_et}

        D_xi = scipy_sparse.csc_matrix(
            (D_xi["elements"], (D_xi["rows"], D_xi["cols"])), shape=(N * M, N * M)
        )
        D_et = scipy_sparse.csc_matrix(
            (D_et["elements"], (D_et["rows"], D_et["cols"])), shape=(N * M, N * M)
        )

        if cube_coordinates:
            if sparse:
                return D_xi, D_et
            else:
                return np.array(D_xi.todense()), np.array(D_et.todense())

        # A scalar gradient is a covector.  Convert its coordinate partials
        # through the exact dual basis of the embedded cubed-sphere surface.
        # This remains well-defined at the projection centre and avoids the
        # 0/0 terms in the historical Ronchi-equation implementation.
        east_coeff, north_coeff, _ = self.surface_geometry()
        Le = (
            scipy_sparse.diags(east_coeff[:, 0]) @ D_xi
            + scipy_sparse.diags(east_coeff[:, 1]) @ D_et
        )
        Ln = (
            scipy_sparse.diags(north_coeff[:, 0]) @ D_xi
            + scipy_sparse.diags(north_coeff[:, 1]) @ D_et
        )
        if sparse:
            return Le, Ln
        else:
            return np.array(Le.todense()), np.array(Ln.todense())

    def surface_gradient_operator(self, stencil_size=1):
        """Return the scalar-to-tangential-gradient linear map."""
        east, north = self.surface_gradient_matrices(stencil_size=stencil_size, sparse=True)
        matrix = scipy_sparse.vstack((east, north), format="csc")
        return as_linear_map(
            matrix,
            input_shape=(self.mesh.size,),
            output_shape=(2, self.mesh.size),
        )

    def surface_divergence_matrix(self, stencil_size=1, *, sparse=True):
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
        stencil_size: int, optional
            Stencil size. Default is 1, in which case derivatives will be calculated
            with a 3-point stencil. With S = 2, a 5-point stencil will be used. etc.
        sparse: bool, optional
            Set to True if you want scipy.sparse matrices instead of dense numpy arrays
        """
        D_xi, D_eta = self.surface_gradient_matrices(
            stencil_size=stencil_size, cube_coordinates=True, sparse=True
        )
        east_coeff, north_coeff, sqrt_g = self.surface_geometry()

        # In curvilinear coordinates,
        # div(V) = 1/sqrt(g) * partial_a(sqrt(g) V^a).
        # The dual-basis coefficients convert geographic east/north vector
        # components into the contravariant components V^xi and V^eta.
        inv_sqrt_g = scipy_sparse.diags(1 / sqrt_g)
        east = inv_sqrt_g @ (
            D_xi @ scipy_sparse.diags(sqrt_g * east_coeff[:, 0])
            + D_eta @ scipy_sparse.diags(sqrt_g * east_coeff[:, 1])
        )
        north = inv_sqrt_g @ (
            D_xi @ scipy_sparse.diags(sqrt_g * north_coeff[:, 0])
            + D_eta @ scipy_sparse.diags(sqrt_g * north_coeff[:, 1])
        )
        result = scipy_sparse.hstack((east, north), format="csc")
        return result if sparse else result.toarray()

    def surface_divergence_operator(self, stencil_size=1):
        """Return the tangential-vector-to-divergence linear map."""
        matrix = self.surface_divergence_matrix(stencil_size=stencil_size, sparse=True)
        return as_linear_map(
            matrix,
            input_shape=(2, self.mesh.size),
            output_shape=(self.mesh.size,),
        )

    def surface_geometry(self):
        """Return geographic dual-basis coefficients and ``sqrt(det(g))``.

        The returned east and north arrays have columns for the xi and eta
        contravariant directions.  They serve both scalar-gradient and vector-
        divergence operators, keeping those two constructions geometrically
        consistent.
        """
        grid = self.mesh
        xi = grid.xi.reshape(-1)
        eta = grid.eta.reshape(-1)

        # Columns 0 and 1 of the inverse coordinate transform are the physical
        # tangent vectors d(position)/dxi and d(position)/deta in local ECEF.
        jacobian_local = cs_vectors.pc(xi, eta, r=grid.radius, block=4, inverse=True)[:, :, :2]
        jacobian_geo = np.einsum("ij,njk->nik", grid.projection.R_local2geo, jacobian_local)
        metric = np.einsum("nki,nkj->nij", jacobian_geo, jacobian_geo)
        dual_basis = np.einsum("nki,nij->nkj", jacobian_geo, np.linalg.inv(metric))

        lon = np.deg2rad(grid.lon.reshape(-1))
        lat = np.deg2rad(grid.lat.reshape(-1))
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

    def interpolate_scalar(self, values, lon, lat):
        """
        Interpolate values of a cell-centred scalar field at the requested
        longitude/latitude locations. Bilinear interpolation uses only the
        nearest four values and therefore requires memory proportional to the
        number of evaluation points, not the product of grid and point counts.

        Parameters
        ----------
        values: array
            2D array (or flattened) of the scalar field as defined on the CS grid
            (dimensions must match)
        lon: array
            array of longitudes [degrees] - must have same shape as lat_
        lat: array
            array of latitudes [degrees] - must have same shape as lon_

        Returns
        -------
        Interpolated values of the 2D scalar field at the desired input (lon, lat)
        locations.
        """
        grid = self.mesh
        lon, lat = np.broadcast_arrays(np.asarray(lon), np.asarray(lat))

        # Remove points outside grid
        inside = grid.contains(lon, lat)
        lon_ = lon[inside]
        lat_ = lat[inside]

        # Get i,j index of each evaluation location, as a float
        binnumber = grid.bin_index(lon_, lat_)
        i = binnumber[0].flatten()
        j = binnumber[1].flatten()
        xi_obs, eta_obs = grid.projection.geographic_to_cube(lon_.flatten(), lat_.flatten())
        xi_grid = grid.xi[i, j]
        eta_grid = grid.eta[i, j]
        i_frac = (eta_obs - eta_grid) / grid.deta
        j_frac = (xi_obs - xi_grid) / grid.dxi
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
        small_i = i <= 0  # due to contains it must also be > -0.5
        i[small_i] = 0.00001
        small_j = j <= 0  # due to contains it must also be > -0.5
        j[small_j] = 0.00001
        large_i = i >= grid.shape[0] - 1  # due to contains it must also be > -0.5
        i[large_i] = grid.shape[0] - 1 - 0.00001
        large_j = j >= grid.shape[1] - 1  # due to contains it must also be > -0.5
        j[large_j] = grid.shape[1] - 1 - 0.00001

        # Indices of the four nodes surrounding each observation/evaluation location
        ifloor = np.floor(i).astype(int)
        iceil = np.ceil(i).astype(int)
        jfloor = np.floor(j).astype(int)
        jceil = np.ceil(j).astype(int)

        # CS coordinates and 1D indices of these points
        xi1 = grid.xi[ifloor, jfloor]
        eta1 = grid.eta[ifloor, jfloor]
        ij1 = grid.flat_index(ifloor, jfloor)
        # xi2 = grid.xi[iceil,jfloor] # We dont need all 4 the xi-eta coords. since their
        # coordinates will be the pairwise similar
        eta2 = grid.eta[iceil, jfloor]
        ij2 = grid.flat_index(iceil, jfloor)
        # xi3 = grid.xi[iceil,jceil]
        # eta3 = grid.eta[iceil,jceil]
        ij3 = grid.flat_index(iceil, jceil)
        xi4 = grid.xi[ifloor, jceil]
        # eta4 = grid.eta[ifloor,jceil]
        ij4 = grid.flat_index(ifloor, jceil)

        # CS coordinates of observations/evaluation locations
        xi_obs, eta_obs = grid.projection.geographic_to_cube(lon_.flatten(), lat_.flatten())
        # Bilinear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
        w1 = (xi4 - xi_obs) * (eta2 - eta_obs) / ((xi4 - xi1) * (eta2 - eta1))  # w11
        w2 = (xi4 - xi_obs) * (eta_obs - eta1) / ((xi4 - xi1) * (eta2 - eta1))  # w12
        w3 = (xi_obs - xi1) * (eta_obs - eta1) / ((xi4 - xi1) * (eta2 - eta1))  # w22
        w4 = (xi_obs - xi1) * (eta2 - eta_obs) / ((xi4 - xi1) * (eta2 - eta1))  # w21

        scalar_values = np.asarray(values).reshape(-1)
        if scalar_values.size != grid.size:
            raise ValueError(
                f"values must contain {grid.size} grid values; got {scalar_values.size}"
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


@dataclass(frozen=True)
class RegionalCSMeshSpec:
    """Versioned, consumer-neutral specification for a regional CS mesh."""

    position: tuple[float, float]
    orientation: tuple[float, float]
    length: float
    width: float
    radius: float
    shape: tuple[int, int] | None = None
    cell_size: tuple[float, float] | None = None
    xi_edges: tuple[float, ...] | None = None
    eta_edges: tuple[float, ...] | None = None
    xi_shift: float = 0.0
    schema: str = REGIONAL_CS_MESH_SCHEMA
    version: int = REGIONAL_CS_MESH_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema != REGIONAL_CS_MESH_SCHEMA:
            raise ValueError(f"Unsupported regional-mesh schema: {self.schema!r}")
        if self.version != REGIONAL_CS_MESH_SCHEMA_VERSION:
            raise ValueError(f"Unsupported regional-mesh schema version: {self.version!r}")

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

        xi_shift = float(self.xi_shift)
        if not np.isfinite(xi_shift):
            raise ValueError("xi_shift must be finite")
        object.__setattr__(self, "xi_shift", xi_shift)

        shape = self._shape(self.shape)
        cell_size = self._cell_size(self.cell_size)
        xi_edges = self._edge_axis("xi_edges", self.xi_edges)
        eta_edges = self._edge_axis("eta_edges", self.eta_edges)
        explicit_edges = xi_edges is not None or eta_edges is not None
        if explicit_edges and (xi_edges is None or eta_edges is None):
            raise ValueError("xi_edges and eta_edges must be provided together")
        mode_count = int(shape is not None) + int(cell_size is not None) + int(explicit_edges)
        if mode_count != 1:
            raise ValueError("Provide exactly one of shape, cell_size, or explicit edges")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(self, "xi_edges", xi_edges)
        object.__setattr__(self, "eta_edges", eta_edges)

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
    def _shape(value):
        if value is None:
            return None
        if len(value) != 2 or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
            for item in value
        ):
            raise TypeError("shape must contain two integer cell counts")
        result = tuple(int(item) for item in value)
        if any(item <= 0 for item in result):
            raise ValueError("shape cell counts must be positive")
        return result

    @staticmethod
    def _cell_size(value):
        if value is None:
            return None
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError) as error:
            raise TypeError("cell_size must contain two numbers") from error
        if len(result) != 2 or not np.isfinite(result).all() or any(item <= 0 for item in result):
            raise ValueError("cell_size must contain two positive finite values")
        return result

    @staticmethod
    def _edge_axis(name, values):
        if values is None:
            return None
        result = tuple(float(value) for value in values)
        array = np.asarray(result)
        if array.size < 2 or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain at least two finite values")
        if not np.all(np.diff(array) > 0):
            raise ValueError(f"{name} must be strictly increasing")
        spacing = np.diff(array)
        if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-15):
            raise ValueError(f"{name} must be uniformly spaced")
        return result

    @classmethod
    def from_mesh(cls, mesh):
        """Create a specification from a canonical mesh."""
        if not isinstance(mesh, RegionalCSMesh):
            raise TypeError("mesh must be a RegionalCSMesh")
        explicit_edges = mesh.requested_shape is None and mesh.requested_cell_size is None
        return cls(
            position=tuple(mesh.projection.position),
            orientation=tuple(mesh.projection.orientation),
            length=mesh.length,
            width=mesh.width,
            radius=mesh.radius,
            shape=mesh.requested_shape,
            cell_size=mesh.requested_cell_size,
            xi_edges=mesh.xi_edges if explicit_edges else None,
            eta_edges=mesh.eta_edges if explicit_edges else None,
            xi_shift=0.0 if explicit_edges else mesh.xi_shift,
        )

    @classmethod
    def from_dict(cls, metadata):
        """Parse the versioned canonical mapping format."""
        if not isinstance(metadata, Mapping):
            raise TypeError("regional-mesh metadata must be a mapping")
        projection = metadata.get("projection")
        if not isinstance(projection, Mapping):
            raise TypeError("projection metadata must be a mapping")
        return cls(
            schema=metadata.get("schema"),
            version=metadata.get("version"),
            position=projection.get("position"),
            orientation=projection.get("orientation"),
            length=metadata.get("length"),
            width=metadata.get("width"),
            radius=metadata.get("radius"),
            shape=metadata.get("shape"),
            cell_size=metadata.get("cell_size"),
            xi_edges=metadata.get("xi_edges"),
            eta_edges=metadata.get("eta_edges"),
            xi_shift=metadata.get("xi_shift", 0.0),
        )

    def to_dict(self):
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
            "radius": self.radius,
            "shape": None if self.shape is None else list(self.shape),
            "cell_size": None if self.cell_size is None else list(self.cell_size),
            "xi_edges": None if self.xi_edges is None else list(self.xi_edges),
            "eta_edges": None if self.eta_edges is None else list(self.eta_edges),
            "xi_shift": self.xi_shift,
        }

    def to_mesh(self):
        """Construct a regional cubed-sphere mesh from this specification."""
        projection = RegionalCSProjection(self.position, self.orientation)
        return RegionalCSMesh(
            projection,
            self.length,
            self.width,
            radius=self.radius,
            shape=self.shape,
            cell_size=self.cell_size,
            xi_edges=self.xi_edges,
            eta_edges=self.eta_edges,
            xi_shift=self.xi_shift,
        )


__all__ = [
    "REGIONAL_CS_MESH_SCHEMA",
    "REGIONAL_CS_MESH_SCHEMA_VERSION",
    "RegionalCSMesh",
    "RegionalCSMeshSpec",
    "RegionalCSOperators",
    "RegionalCSProjection",
]
