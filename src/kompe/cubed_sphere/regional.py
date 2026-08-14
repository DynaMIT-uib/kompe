"""Regional geometry and operators on one rotated cubed-sphere face.

``RegionalCSProjection`` rotates the requested geographic centre and orientation
onto the north face of the shared global cubed-sphere chart. ``RegionalCSMesh``
discretizes a bounded patch of that face, and ``RegionalCSOperators`` applies the
corresponding non-orthogonal metric, interpolation, gradient, and divergence.

The geometry follows C. Ronchi, R. Iacono, and P. S. Paolucci, *The Cubed
Sphere: A New Method for the Solution of Partial Differential Equations in
Spherical Geometry*, J. Comput. Phys. 124 (1996), 93–114.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy import sparse as scipy_sparse

from kompe.cubed_sphere import cs_coordinates
from kompe.grid import SphericalGrid
from kompe.math import as_linear_map, content_fingerprint
from kompe.mesh import StructuredSurfaceMesh

from . import cs_vectors, diffutils, spherical

_DATA_PATH = Path(__file__).resolve().parents[1] / "data"

REGIONAL_CS_MESH_SCHEMA = "kompe.regional_cs_mesh"
REGIONAL_CS_MESH_SCHEMA_VERSION = 1
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


def _interpolation_axis(position, first_center, spacing, count):
    """Return neighbouring indices and fractions on one uniform cell-centre axis."""
    if count == 1:
        index = np.zeros(position.size, dtype=int)
        return index, index, np.zeros(position.size)
    coordinate = (position - first_center) / spacing
    lower = np.clip(np.floor(coordinate).astype(int), 0, count - 2)
    return lower, lower + 1, coordinate - lower


def _coordinate_pair(name, values):
    """Return a pair of finite floating-point coordinates."""
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain two numbers") from error
    if len(result) != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite numbers")
    return result


def _mesh_shape(value):
    """Return a positive ``(n_eta, n_xi)`` mesh shape."""
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


def _positive_cell_size(name, value):
    """Return one positive finite physical cell size."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a number") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _cell_size_pair(value):
    """Normalize the persisted ``(eta, xi)`` cell-size representation."""
    if value is None:
        return None
    try:
        eta_cell_size, xi_cell_size = value
    except (TypeError, ValueError) as error:
        raise TypeError("cell_size must contain two numbers") from error
    return (
        _positive_cell_size("eta cell size", eta_cell_size),
        _positive_cell_size("xi cell size", xi_cell_size),
    )


def _uniform_edge_axis(name, values):
    """Return one strictly increasing, uniformly spaced edge axis."""
    if values is None:
        return None
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numbers") from error
    array = np.asarray(result)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain at least two finite values")
    if not np.all(np.diff(array) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    spacing = np.diff(array)
    if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-15):
        raise ValueError(f"{name} must be uniformly spaced")
    return result


class RegionalCSProjection:
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
        self.z = np.array(
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ]
        )

        # On the north face, increasing xi follows the local Cartesian y axis.
        self.y = spherical.enu_to_ecef(
            orientation_enu,
            np.array(self.lon0),
            np.array(self.lat0),
        ).flatten()

        # Complete the right-handed local Cartesian frame.
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
        """Convert from geocentric coordinates to cube coords (xi, eta)

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
        xi, eta, _ = cs_coordinates.geo_to_cube(
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
        """Convert from cube coordinates (xi, eta) to geocentric (lon, lat)

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
        """Convert from geocentric coordinates to local coordinates

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
        return _rotate_spherical_coordinates(lon, lat, self.R_geo2local)

    def local_to_geographic(self, lon, lat):
        """Convert from local coordinates to geocentric coordinates

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
        return _rotate_spherical_coordinates(lon, lat, self.R_local2geo)

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
        lon, lat = map(np.ravel, np.broadcast_arrays(lon, lat))
        geographic_lon, geographic_lat = self.local_to_geographic(lon, lat)
        local_east = spherical.enu_to_ecef(np.tile((1.0, 0.0, 0.0), (lon.size, 1)), lon, lat)
        local_north = spherical.enu_to_ecef(np.tile((0.0, 1.0, 0.0), (lon.size, 1)), lon, lat)
        geographic_east = spherical.ecef_to_enu(
            np.einsum("ij,nj->ni", self.R_local2geo, local_east),
            geographic_lon,
            geographic_lat,
        )[:, :2]
        geographic_north = spherical.ecef_to_enu(
            np.einsum("ij,nj->ni", self.R_local2geo, local_north),
            geographic_lon,
            geographic_lat,
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
        geographic_ecef = spherical.enu_to_ecef(
            np.column_stack((east, north, np.zeros_like(east))),
            lon,
            lat,
        )
        local_ecef = np.einsum("ij,nj->ni", self.R_geo2local, geographic_ecef)
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
        geographic_ecef = np.einsum("ij,nj->ni", self.R_local2geo, local_ecef)
        geographic = spherical.ecef_to_enu(geographic_ecef, lon, lat)
        return lon, lat, geographic[:, 0], geographic[:, 1]

    def projected_coastlines(self, resolution="50m"):
        """Generate coastlines in projected coordinates"""
        coastlines = np.load(_DATA_PATH / f"coastlines_{resolution}.npz")
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
        xi, eta, dxi, deta, radius = np.broadcast_arrays(xi, eta, dxi, deta, radius)
        metric = cs_coordinates.surface_metric_tensor(xi, eta, r=radius).reshape(xi.shape + (2, 2))

        dlxi = np.sqrt(metric[..., 0, 0]) * dxi
        dleta = np.sqrt(metric[..., 1, 1]) * deta
        area_scale = np.sqrt(metric[..., 0, 0] * metric[..., 1, 1] - metric[..., 0, 1] ** 2)
        dS = area_scale * dxi * deta

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
        xi_cell_size=None,
        eta_cell_size=None,
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
            Persisted ``(eta, xi)`` form of the physical cell sizes. Interactive
            code should prefer the explicitly named cell-size parameters below.
        xi_cell_size, eta_cell_size: float, optional
            Target physical cell sizes parallel and perpendicular to the projection
            orientation, respectively. The final uniform spacing is adjusted slightly
            so the requested extent is exact. Both values must be provided together.
        xi_edges, eta_edges: array-like, optional
            Exact uniformly spaced computational-coordinate edges in radians. Prefer
            :meth:`from_edges` when constructing a mesh this way.
        xi_shift: float, optional
            Physical displacement along the xi axis, in the same units as ``radius``.

        Notes
        -----
        Provide exactly one construction mode: ``shape``, one cell-size form, or
        both explicit edge arrays. Explicit physical-axis names avoid depending on
        NumPy's ``(eta, xi)`` array-axis order when specifying resolution.

        """
        if not isinstance(projection, RegionalCSProjection):
            raise TypeError("projection must be a RegionalCSProjection")
        for name, value in (("length", length), ("width", width), ("radius", radius)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        shape = _mesh_shape(shape)
        cell_size = _cell_size_pair(cell_size)
        xi_cell_size = _positive_cell_size("xi_cell_size", xi_cell_size)
        eta_cell_size = _positive_cell_size("eta_cell_size", eta_cell_size)
        named_cell_sizes = xi_cell_size is not None or eta_cell_size is not None
        if named_cell_sizes and (xi_cell_size is None or eta_cell_size is None):
            raise ValueError("xi_cell_size and eta_cell_size must be provided together")
        if cell_size is not None and named_cell_sizes:
            raise ValueError("Use either cell_size or named xi/eta cell sizes, not both")
        if named_cell_sizes:
            cell_size = (eta_cell_size, xi_cell_size)
        xi_edges = _uniform_edge_axis("xi_edges", xi_edges)
        eta_edges = _uniform_edge_axis("eta_edges", eta_edges)
        explicit_edges = xi_edges is not None or eta_edges is not None
        if explicit_edges and (xi_edges is None or eta_edges is None):
            raise ValueError("xi_edges and eta_edges must be provided together")
        mode_count = int(shape is not None) + int(cell_size is not None) + int(explicit_edges)
        if mode_count != 1:
            raise ValueError(
                "Provide exactly one resolution mode: shape, named xi/eta cell sizes, "
                "persisted cell_size, or explicit edges"
            )
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
            n_eta = max(1, round(eta_span / np.arctan(eta_cell_size / self.radius)))
            n_xi = max(1, round(xi_span / np.arctan(xi_cell_size / self.radius)))
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

        # lon, lat coordinates of cell corners:
        self.lon_mesh, self.lat_mesh = self.projection.cube_to_geographic(
            self.xi_mesh, self.eta_mesh
        )

        # xi, eta coordinates of grid points (cell centers):
        self.xi = self.xi_mesh[0:-1, 0:-1] + self.dxi / 2
        self.eta = self.eta_mesh[0:-1, 0:-1] + self.deta / 2

        # geocentric lon, lat [deg] of grid points:
        self.lon, self.lat = self.projection.cube_to_geographic(self.xi, self.eta)

        # set size and shape
        self._shape = tuple(int(length) for length in self.lat.shape)

        # calculate cell area
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
            "_cell_areas",
        ):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            setattr(self, name, values)
        self.validate_mesh_metadata()

    @classmethod
    def from_edges(cls, projection, xi_edges, eta_edges, *, radius):
        """Construct a mesh from exact computational-coordinate edges.

        This is the natural boundary for saved geometry and for algorithms that
        derive one mesh from another. Ordinary interactive use should normally
        specify physical ``length`` and ``width`` together with either ``shape``
        or explicit physical cell sizes.
        """
        xi_edges = _uniform_edge_axis("xi_edges", xi_edges)
        eta_edges = _uniform_edge_axis("eta_edges", eta_edges)
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be a positive finite number")
        return cls(
            projection,
            radius * np.tan(xi_edges[-1] - xi_edges[0]),
            radius * np.tan(eta_edges[-1] - eta_edges[0]),
            radius=radius,
            xi_edges=xi_edges,
            eta_edges=eta_edges,
        )

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
        """Summarize the regional mesh for interactive inspection."""
        centre_lon, centre_lat = self.projection.cube_to_geographic(
            (self.xi_min + self.xi_max) / 2,
            (self.eta_min + self.eta_max) / 2,
        )
        return (
            f"RegionalCSMesh(shape={self.shape}, center=({float(centre_lon):.1f}, "
            f"{float(centre_lat):.1f}), radius={self.radius:g})"
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
        lon, lat = map(np.ravel, np.broadcast_arrays(lon, lat))
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
        lon, lat = map(np.ravel, np.broadcast_arrays(lon, lat))
        xi, eta = self.projection.geographic_to_cube(lon, lat)

        xi_edges, eta_edges = self.xi_mesh[0, :], self.eta_mesh[:, 0]

        i = np.digitize(eta, eta_edges) - 1
        j = np.digitize(xi, xi_edges) - 1

        iii = ~self.contains(lon, lat)  # points not in grid
        i[iii] = -1
        j[iii] = -1

        return (i, j)

    def contains(self, lon, lat, *, margin_cells=0):
        """
        Determine if lon, lat are inside grid boundaries or not.

        Parameters
        ----------
        lon: array
            array of longitudes [degrees] - must have same shape as lat
        lat: array
            array of latitudes [degrees] - must have same shape as lon
        margin_cells: float, optional
            Number of cell widths by which to extend the boundary. Negative
            values contract it.

        Returns
        -------
        array of bools with shape of lon and lat
        """
        lon, lat = np.broadcast_arrays(lon, lat)
        shape = lon.shape
        xi, eta = self.projection.geographic_to_cube(lon.reshape(-1), lat.reshape(-1))
        ximin = self.xi_min - margin_cells * self.dxi
        ximax = self.xi_max + margin_cells * self.dxi
        etamin = self.eta_min - margin_cells * self.deta
        etamax = self.eta_max + margin_cells * self.deta

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

    def coordinate_derivative_matrices(self, stencil_size=1, *, sparse=True):
        """Return partial-derivative matrices with respect to xi and eta.

        Parameters
        ----------
        stencil_size: int, optional
            Stencil size. Default is 1, in which case derivatives will be calculated
            with a 3-point stencil. With S = 2, a 5-point stencil will be used. etc.
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
        deta = grid.deta
        N = grid.n_eta
        M = grid.n_xi

        D_xi = {"rows": [], "cols": [], "elements": []}
        D_eta = {"rows": [], "cols": [], "elements": []}

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
            D_eta["rows"].append(grid.flat_index(i_dx, j_dx))
            D_eta["cols"].append(grid.flat_index(i_dx + points[ll], j_dx))
            D_eta["elements"].append(np.full(i_dx.size, coefficients[ll] / deta))

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
                D_eta["rows"].append(grid.flat_index(i_dx, j_dx))
                D_eta["cols"].append(grid.flat_index(i_dx + points[ll], j_dx))
                D_eta["elements"].append(np.full(i_dx.size, coefficients[ll] / deta))

                D_xi["rows"].append(grid.flat_index(i_dy, j_dy))
                D_xi["cols"].append(grid.flat_index(i_dy, j_dy + points[ll]))
                D_xi["elements"].append(np.full(i_dy.size, coefficients[ll] / dxi))

            # RIGHT
            points = np.r_[-S : kk + 1 : 1]
            coefficients = diffutils.stencil(points, order=1)
            i_dx, j_dx = ii[:, -(kk + 1)], jj[:, -(kk + 1)]
            i_dy, j_dy = ii.T[:, -(kk + 1)], jj.T[:, -(kk + 1)]

            for ll in range(len(points)):
                D_eta["rows"].append(grid.flat_index(i_dx, j_dx))
                D_eta["cols"].append(grid.flat_index(i_dx + points[ll], j_dx))
                D_eta["elements"].append(np.full(i_dx.size, coefficients[ll] / deta))

                D_xi["rows"].append(grid.flat_index(i_dy, j_dy))
                D_xi["cols"].append(grid.flat_index(i_dy, j_dy + points[ll]))
                D_xi["elements"].append(np.full(i_dy.size, coefficients[ll] / dxi))

        D_xi = {key: np.hstack(D_xi[key]) for key in D_xi}
        D_eta = {key: np.hstack(D_eta[key]) for key in D_eta}

        D_xi = scipy_sparse.csc_matrix(
            (D_xi["elements"], (D_xi["rows"], D_xi["cols"])), shape=(N * M, N * M)
        )
        D_eta = scipy_sparse.csc_matrix(
            (D_eta["elements"], (D_eta["rows"], D_eta["cols"])), shape=(N * M, N * M)
        )

        if sparse:
            return D_xi, D_eta
        return D_xi.toarray(), D_eta.toarray()

    def surface_gradient_matrices(self, stencil_size=1, *, sparse=True):
        """Return scalar-gradient matrices in ``(theta, phi)`` order.

        ``theta`` points south and ``phi`` points east. The matrices act on
        flattened cell-centred scalar values.
        """
        D_xi, D_eta = self.coordinate_derivative_matrices(
            stencil_size=stencil_size,
            sparse=True,
        )

        # A scalar gradient is a covector.  Convert its coordinate partials
        # through the exact dual basis of the embedded cubed-sphere surface.
        # This remains well-defined at the projection centre and avoids the
        # 0/0 terms in the historical Ronchi-equation implementation.
        theta_coeff, phi_coeff, _ = self._surface_geometry
        D_theta = (
            scipy_sparse.diags(theta_coeff[:, 0]) @ D_xi
            + scipy_sparse.diags(theta_coeff[:, 1]) @ D_eta
        )
        D_phi = (
            scipy_sparse.diags(phi_coeff[:, 0]) @ D_xi
            + scipy_sparse.diags(phi_coeff[:, 1]) @ D_eta
        )
        if sparse:
            return D_theta, D_phi
        return D_theta.toarray(), D_phi.toarray()

    def surface_gradient_operator(self, stencil_size=1):
        """Return the scalar-to-tangential-gradient linear map."""
        theta, phi = self.surface_gradient_matrices(stencil_size=stencil_size, sparse=True)
        matrix = scipy_sparse.vstack((theta, phi), format="csc")
        return as_linear_map(
            matrix,
            input_shape=(self.mesh.size,),
            output_shape=(2, self.mesh.size),
        )

    def surface_divergence_matrix(self, stencil_size=1, *, sparse=True):
        """
        Calculate the matrix that produces the divergence of a vector field

        The returned N x 2N matrix operates on a 1D array that represents a
        vector field. The array must be of length 2N, where N is the number
        of grid cells. The first N elements are the southward ``theta``
        components and the last N are the eastward ``phi`` components.

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
        D_xi, D_eta = self.coordinate_derivative_matrices(
            stencil_size=stencil_size,
            sparse=True,
        )
        theta_coeff, phi_coeff, sqrt_g = self._surface_geometry

        # In curvilinear coordinates,
        # div(V) = 1/sqrt(g) * partial_a(sqrt(g) V^a).
        # The dual-basis coefficients convert spherical theta/phi vector
        # components into the contravariant components V^xi and V^eta.
        inv_sqrt_g = scipy_sparse.diags(1 / sqrt_g)
        theta = inv_sqrt_g @ (
            D_xi @ scipy_sparse.diags(sqrt_g * theta_coeff[:, 0])
            + D_eta @ scipy_sparse.diags(sqrt_g * theta_coeff[:, 1])
        )
        phi = inv_sqrt_g @ (
            D_xi @ scipy_sparse.diags(sqrt_g * phi_coeff[:, 0])
            + D_eta @ scipy_sparse.diags(sqrt_g * phi_coeff[:, 1])
        )
        result = scipy_sparse.hstack((theta, phi), format="csc")
        return result if sparse else result.toarray()

    def surface_divergence_operator(self, stencil_size=1):
        """Return the tangential-vector-to-divergence linear map."""
        matrix = self.surface_divergence_matrix(stencil_size=stencil_size, sparse=True)
        return as_linear_map(
            matrix,
            input_shape=(2, self.mesh.size),
            output_shape=(self.mesh.size,),
        )

    @cached_property
    def _surface_geometry(self):
        """Return spherical dual-basis coefficients and ``sqrt(det(g))``.

        The returned theta and phi arrays have columns for the xi and eta
        contravariant directions.  They serve both scalar-gradient and vector-
        divergence operators, keeping those two constructions geometrically
        consistent.
        """
        grid = self.mesh
        xi = grid.xi.reshape(-1)
        eta = grid.eta.reshape(-1)

        # Rows 0 and 1 are the dual basis vectors grad(xi) and grad(eta)
        # in local Cartesian coordinates.
        dual_basis_local = cs_vectors._cartesian_to_cube_matrix(
            xi,
            eta,
            r=grid.radius,
            block=_NORTH_FACE,
        )[:, :2, :].transpose(0, 2, 1)
        dual_basis = np.einsum(
            "ij,njk->nik",
            grid.projection.R_local2geo,
            dual_basis_local,
        )
        metric = cs_coordinates.surface_metric_tensor(xi, eta, r=grid.radius)

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
        return -north_coeff, east_coeff, sqrt_g

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
        scalar_values = np.asarray(values).reshape(-1)
        if scalar_values.size != grid.size:
            raise ValueError(
                f"values must contain {grid.size} grid values; got {scalar_values.size}"
            )
        xi, eta = grid.projection.geographic_to_cube(lon.reshape(-1), lat.reshape(-1))
        coordinate_tolerance = 32 * np.finfo(float).eps
        inside = (
            (xi >= grid.xi_min - coordinate_tolerance)
            & (xi <= grid.xi_max + coordinate_tolerance)
            & (eta >= grid.eta_min - coordinate_tolerance)
            & (eta <= grid.eta_max + coordinate_tolerance)
        )
        xi_inside = xi[inside]
        eta_inside = eta[inside]
        i0, i1, eta_fraction = _interpolation_axis(
            eta_inside,
            grid.eta[0, 0],
            grid.deta,
            grid.n_eta,
        )
        j0, j1, xi_fraction = _interpolation_axis(
            xi_inside,
            grid.xi[0, 0],
            grid.dxi,
            grid.n_xi,
        )
        field = scalar_values.reshape(grid.shape)
        interpolated_inside = (
            (1 - eta_fraction) * (1 - xi_fraction) * field[i0, j0]
            + eta_fraction * (1 - xi_fraction) * field[i1, j0]
            + eta_fraction * xi_fraction * field[i1, j1]
            + (1 - eta_fraction) * xi_fraction * field[i0, j1]
        )
        interpolated = np.full(
            lat.size,
            np.nan,
            dtype=np.result_type(interpolated_inside, float),
        )
        interpolated[inside] = interpolated_inside

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

        position = _coordinate_pair("position", self.position)
        orientation = _coordinate_pair("orientation", self.orientation)
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

        shape = _mesh_shape(self.shape)
        cell_size = _cell_size_pair(self.cell_size)
        xi_edges = _uniform_edge_axis("xi_edges", self.xi_edges)
        eta_edges = _uniform_edge_axis("eta_edges", self.eta_edges)
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
        if self.xi_edges is not None:
            # Version-1 metadata allowed a shift together with stored edges.
            # Preserve that exact persisted meaning at the schema boundary;
            # RegionalCSMesh.from_edges() treats its edges as final geometry.
            return RegionalCSMesh(
                projection,
                self.length,
                self.width,
                radius=self.radius,
                xi_edges=self.xi_edges,
                eta_edges=self.eta_edges,
                xi_shift=self.xi_shift,
            )
        mesh_resolution = {"shape": self.shape}
        if self.cell_size is not None:
            eta_cell_size, xi_cell_size = self.cell_size
            mesh_resolution = {
                "xi_cell_size": xi_cell_size,
                "eta_cell_size": eta_cell_size,
            }
        return RegionalCSMesh(
            projection,
            self.length,
            self.width,
            radius=self.radius,
            xi_shift=self.xi_shift,
            **mesh_resolution,
        )


__all__ = [
    "REGIONAL_CS_MESH_SCHEMA",
    "REGIONAL_CS_MESH_SCHEMA_VERSION",
    "RegionalCSMesh",
    "RegionalCSMeshSpec",
    "RegionalCSOperators",
    "RegionalCSProjection",
]
