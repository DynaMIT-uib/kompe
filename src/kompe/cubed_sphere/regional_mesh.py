"""Structured mesh geometry on a regional cubed-sphere face."""

from collections.abc import Mapping
from functools import cached_property

import numpy as np

from kompe.cubed_sphere.regional_projection import RegionalCSProjection
from kompe.grid import SphericalGrid
from kompe.math import backend_context, content_fingerprint
from kompe.mesh import StructuredSurfaceMesh


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


class RegionalCSMesh(StructuredSurfaceMesh):
    """Structured bounded mesh on one rotated cubed-sphere face."""

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

        Create a regular grid in xi, eta coordinates. ``length`` is the extent
        along the projection's xi axis; ``width`` is the extent along eta. The
        grid is centred at ``projection.position``.

        Parameters
        ----------
        projection : RegionalCSProjection
            Coordinate projection that sets the mesh centre and orientation.
        length : float
            Physical extent along xi, parallel to ``projection.orientation``.
        width : float
            Physical extent along eta, perpendicular to
            ``projection.orientation``.
        radius : float
            Radius of the sphere, in the same units as the dimensions and any
            cell sizes. It is required so that the mesh never
            silently assumes kilometres or metres.
        shape : tuple of int, optional
            Number of cells along the ``(eta, xi)`` axes.
        cell_size : tuple of float, optional
            Persisted ``(eta, xi)`` form of the physical cell sizes. Interactive
            code should prefer the explicitly named cell-size parameters below.
        xi_cell_size, eta_cell_size : float, optional
            Target physical cell sizes parallel and perpendicular to the projection
            orientation, respectively. The final uniform spacing is adjusted slightly
            so the requested extent is exact. Both values must be provided together.
        xi_edges, eta_edges : array-like, optional
            Exact uniformly spaced computational-coordinate edges in radians. Prefer
            :meth:`from_edges` when constructing a mesh this way.
        xi_shift : float, optional
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

        # Number of cells along the eta and xi array axes.
        self.n_eta, self.n_xi = len(eta_edge) - 1, len(xi_edge) - 1

        # size of grid cells in xi, eta coordinates:
        self.dxi = xi_edge[1] - xi_edge[0]
        self.deta = eta_edge[1] - eta_edge[0]

        # xi, eta coordinates of cell corners:
        self.xi_mesh, self.eta_mesh = np.meshgrid(xi_edge, eta_edge, indexing="xy")

        # Mesh coordinates and cell areas are immutable host geometry. Keep
        # their construction independent of the active field-array backend.
        with backend_context("numpy"):
            self.lon_mesh, self.lat_mesh = self.projection.cube_to_geographic(
                self.xi_mesh, self.eta_mesh
            )

            # xi, eta coordinates of grid points (cell centers):
            self.xi = self.xi_mesh[0:-1, 0:-1] + self.dxi / 2
            self.eta = self.eta_mesh[0:-1, 0:-1] + self.deta / 2

            # geocentric lon, lat [deg] of grid points:
            self.lon, self.lat = self.projection.cube_to_geographic(self.xi, self.eta)
            cell_areas = self.projection.differential_elements(
                self.xi, self.eta, self.dxi, self.deta, radius=self.radius
            )[2]

        # set size and shape
        self._shape = tuple(int(length) for length in self.lat.shape)

        # calculate cell area
        self._cell_areas = cell_areas

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
        from kompe.cubed_sphere.regional_operators import RegionalCSOperators

        return RegionalCSOperators(self)

    def to_spec(self):
        """Return a versioned, JSON-serializable grid specification."""
        from kompe.cubed_sphere.regional_mesh_spec import RegionalCSMeshSpec

        return RegionalCSMeshSpec.from_mesh(self)

    @classmethod
    def from_spec(cls, spec):
        """Construct a grid from a :class:`RegionalCSMeshSpec` or mapping."""
        from kompe.cubed_sphere.regional_mesh_spec import RegionalCSMeshSpec

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

        Parameters
        ----------
        flat_index : array-like of int
            Flattened cell indices.

        Returns
        -------
        Two 1D arrays, first containing the i indices, second the j indices
        Same length (N) as input parameter.

        """
        return np.unravel_index(flat_index, self.shape)

    def count_points(self, lon, lat, **kwargs):
        """Count the number of points in each grid cell.

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
        """Find the cell index ``(i, j)`` for each ``(lon, lat)`` pair.

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
        lon : array
            array of longitudes [degrees] - must have same shape as lat
        lat : array
            array of latitudes [degrees] - must have same shape as lon
        margin_cells : float, optional
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
        """Yield grid boundaries for plotting.

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


__all__ = ["RegionalCSMesh"]
