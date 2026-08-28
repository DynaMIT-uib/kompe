"""Plot regional cubed-sphere data on an existing Matplotlib axis."""

from pathlib import Path

import numpy as np

_DATA_PATH = Path(__file__).resolve().parents[1] / "data"


class RegionalCSPlotter:
    """Draw geographic data and reference grids on a regional CS mesh.

    The plotting methods mirror the corresponding Matplotlib calls but accept
    longitude and latitude, which are projected onto ``mesh`` before drawing.
    """

    def __init__(self, ax, mesh, **kwargs):
        """Attach a regional cubed-sphere plotter to ``ax``.

        Parameters
        ----------
        ax : matplotlib.AxesSubplot
            Matplotlib axes on which to draw.
        mesh : kompe.RegionalCSMesh
            Regional cubed-sphere mesh to draw.
        **kwargs : dict, optional
            Keywords to control grid lines.
            In addition to Line2D properties, the following can be specified:
            gridtype : str or None
                Determines which grid lines that are added to the csplot
                'geo' adds a geographic lon,lat grid
                'km' adds gridlines with equal physical distance in km
                'cs' adds a cubed sphere xi,eta grid
                Default is None (no grid)
            lt : bool
                Local-time grid labels are not implemented. Passing True raises
                ``NotImplementedError``.
            lat_levels : array_like
                Latitudes at which to draw parallels.
            lat_res : int
                Latitude spacing when ``lat_levels`` is omitted.
            lon_levels : array_like
                Longitudes at which to draw meridians.
            lon_res : int
                Longitude spacing when ``lon_levels`` is omitted.
            km_res : int
                Resolution of 'km' grid. If not provided, default values are used.

        """
        self.ax = ax
        self.mesh = mesh
        self.ax.set_aspect("equal")

        # set ax limits
        self.ax.set_xlim((self.mesh.xi_min, self.mesh.xi_max))
        self.ax.set_ylim((self.mesh.eta_min, self.mesh.eta_max))

        # Select gridtype
        gridtype = kwargs.pop("gridtype", None)
        if gridtype not in {None, "geo", "km", "cs"}:
            raise ValueError("gridtype must be 'geo', 'km', 'cs', or None.")

        # Longitude or local time
        lt = bool(kwargs.pop("lt", False))
        if lt:
            raise NotImplementedError("Local-time grid labels are not implemented.")

        # Add grid
        if gridtype is not None:
            kwargs.setdefault("linewidth", 0.5)
            kwargs.setdefault("color", "lightgrey")

            # Add the selected grid
            if gridtype == "cs":
                self.ax.set_xlabel("$\\xi$")
                self.ax.set_ylabel("$\\eta$")
                self.ax.grid(**kwargs)
            elif gridtype == "km":
                km_res = kwargs.pop(
                    "km_res",
                    np.round(self.mesh.radius * (self.mesh.xi_max + self.mesh.eta_max) // 5, -2),
                )
                self.add_km_grid(km_res, **kwargs)
            else:
                if "lat_levels" in kwargs:
                    lat_levels = kwargs.pop("lat_levels")
                else:
                    lat_levels = np.arange(-90, 90, kwargs.pop("lat_res", 10))[1:]
                if "lon_levels" in kwargs:
                    lon_levels = kwargs.pop("lon_levels")
                else:
                    lon_levels = np.arange(0, 360, kwargs.pop("lon_res", 30))
                self.add_spherical_grid(
                    lat_levels=lat_levels,
                    lon_levels=lon_levels,
                    gridtype=gridtype,
                    lt=lt,
                    **kwargs,
                )

        # Remove ticks and tickmarks
        if gridtype != "cs":
            self.ax.xaxis.set_tick_params(labelbottom=False)
            self.ax.yaxis.set_tick_params(labelleft=False)

            self.ax.set_xticks([])
            self.ax.set_yticks([])

    def add_spherical_grid(
        self,
        lat_levels=np.r_[-80:90:10],
        lon_levels=np.r_[0:360:30],
        gridtype="geo",
        lt=False,
        **kwargs,
    ):
        """Draw a geographic longitude/latitude grid.

        Parameters
        ----------
        lat_levels : array_like, optional
            Array with location of latitudinal parallels. The default is np.r_[-80:90:10].
        lon_levels : array_like, optional
            Array with location of longitudinal meridians. The default is np.r_[0:360:30].
        gridtype : str, optional
            Which coordinate system to add. The default is 'geo'.
        lt : bool, optional
            Local-time grid labels are not implemented. Passing True raises
            ``NotImplementedError``.
        **kwargs : dict
            Line2D properties.

        """
        if gridtype != "geo":
            raise ValueError("Only geographic spherical grids are implemented.")
        if lt:
            raise NotImplementedError("Local-time grid labels are not implemented.")

        # Latitudinal parallels

        lon = np.linspace(0, 360, 361) % 360  # Longitidunal locations

        # Convert to cs coordinates
        xi, eta = self.mesh.projection.geographic_to_cube(*np.meshgrid(lon, lat_levels))

        # Plot the grid lines
        self.ax.plot(xi.T, eta.T, **kwargs)

        ## Longitudinal meridians
        lat = np.linspace(-90, 90, 181)  # Latitudinal locations

        # Convert to cs coordinates
        xi, eta = self.mesh.projection.geographic_to_cube(*np.meshgrid(lon_levels, lat))

        # Plot the grid
        self.ax.plot(xi, eta, **kwargs)

        # Minimum "length" of grid line from tick t be plotted
        count_min = self.mesh.radius * (self.mesh.xi_max + self.mesh.eta_max) // 300

        # Add latitudinal ticks
        iii = self.mesh.contains(*np.meshgrid(lon, lat_levels))  # points in csgrid
        lon_mean = circular_mean_degrees(
            np.where(~iii, np.nan, lon[None, :]), axis=1
        )  # mean of lon grid lines
        lon_count = np.sum(iii, axis=1)  # "length" of grid lines
        lon_res = np.mean(np.diff(lon_levels))  # Distance between meridians
        lon_pos = (
            lon_mean // lon_res * lon_res + lon_res / 2
        )  # Move tick location from mean to between meridians

        # Add the latitudinal ticks
        for x, y in zip(
            lon_pos[lon_count > count_min],
            lat_levels[lon_count > count_min],
            strict=True,
        ):
            self.text(x, y, str(int(y)), horizontalalignment="center", verticalalignment="center")

        iii = self.mesh.contains(*np.meshgrid(lon_levels, lat))  # points in csgrid
        lat_mean = np.nanmean(
            np.where(~iii, np.nan, lat[:, None]), axis=0
        )  # mean of lat grid lines
        lat_count = np.sum(iii, axis=0)  # "length" of grid lines
        lat_res = np.mean(np.diff(lat_levels))  # Distance between parallels
        lat_pos = (
            lat_mean // lat_res * lat_res + lat_res / 2
        )  # Move ticks from mean to between parallels

        # Add the longitudinal ticks
        for x, y in zip(
            lon_levels[lat_count > count_min],
            lat_pos[lat_count > count_min],
            strict=True,
        ):
            self.text(x, y, str(int(x)), horizontalalignment="center", verticalalignment="center")

    def add_km_grid(self, resolution, **kwargs):
        """Draw grid lines separated by ``resolution`` kilometres.

        Parameters
        ----------
        resolution : float
            Distance between the grid lines in km.
        **kwargs : 2D line properties.
            Passed to matplotlib.pyplot.plot

        """
        csres = 0.005

        # xi gridlines
        eta = np.arange(self.mesh.eta_min * 2, self.mesh.eta_max * 2, csres)
        # xi(eta) for xi>0
        xi = np.arange(0, self.mesh.xi_max * 1.1, csres)

        diff = self.mesh.projection.differential_elements(
            *np.meshgrid(xi, eta), csres, 0, radius=self.mesh.radius
        )[0]
        diff[:, 0] = 0

        xi_pos = []
        for i in range(len(eta)):
            xi_pos.append(
                np.interp(
                    np.arange(0, self.mesh.length / 2, resolution), np.cumsum(diff[i, :]), xi
                )
            )
        xi_pos = np.array(xi_pos)

        # xi(eta) for xi<0
        xi = np.arange(0, self.mesh.xi_min * 1.1, -csres)

        diff = self.mesh.projection.differential_elements(
            *np.meshgrid(xi, eta), -csres, 0, radius=self.mesh.radius
        )[0]
        diff[:, 0] = 0

        xi_neg = []
        for i in range(len(eta)):
            xi_neg.append(
                np.interp(
                    np.arange(-resolution, -self.mesh.length / 2, -resolution)[::-1],
                    np.cumsum(diff[i, :])[::-1],
                    xi[::-1],
                )
            )
        xi_neg = np.array(xi_neg)

        self.ax.plot(np.hstack((xi_neg, xi_pos)), eta, **kwargs)

        # tickmarks
        idx = (np.abs(eta - self.mesh.eta_min)).argmin()
        xi_tick = np.hstack((xi_neg, xi_pos))[idx, :]
        eta_tick = self.mesh.eta_min * 1.05
        km_tick = np.concatenate(
            (
                np.arange(-resolution, -self.mesh.length / 2, -resolution)[::-1],
                np.arange(0, self.mesh.length / 2, resolution),
            )
        )

        ind = (xi_tick >= self.mesh.xi_min) & (xi_tick <= self.mesh.xi_max)
        xi_tick = xi_tick[ind]
        km_tick = km_tick[ind]
        for x, label in zip(xi_tick, km_tick, strict=True):
            self.ax.text(
                x,
                eta_tick,
                str(int(label)),
                horizontalalignment="center",
                verticalalignment="top",
            )

        # Eta grid lines

        xi = np.arange(self.mesh.xi_min * 2, self.mesh.xi_max * 2, csres)
        # eta(xi) for eta>0
        eta = np.arange(0, self.mesh.eta_max * 1.1, csres)

        diff = self.mesh.projection.differential_elements(
            *np.meshgrid(xi, eta), 0, csres, radius=self.mesh.radius
        )[1].T
        diff[:, 0] = 0

        eta_pos = []
        for i in range(len(xi)):
            eta_pos.append(
                np.interp(
                    np.arange(0, self.mesh.width / 2, resolution), np.cumsum(diff[i, :]), eta
                )
            )
        eta_pos = np.array(eta_pos)

        # eta(xi) for eta<0
        eta = np.arange(0, self.mesh.eta_min * 1.1, -csres)

        diff = self.mesh.projection.differential_elements(
            *np.meshgrid(xi, eta), 0, -csres, radius=self.mesh.radius
        )[1].T
        diff[:, 0] = 0

        eta_neg = []
        for i in range(len(xi)):
            eta_neg.append(
                np.interp(
                    np.arange(-resolution, -self.mesh.width / 2, -resolution)[::-1],
                    np.cumsum(diff[i, :])[::-1],
                    eta[::-1],
                )
            )
        eta_neg = np.array(eta_neg)

        self.ax.plot(xi, np.hstack((eta_neg, eta_pos)), **kwargs)

        # tickmarks
        xi_tick = self.mesh.xi_min * 1.05
        idx = (np.abs(xi - self.mesh.xi_min)).argmin()
        eta_tick = np.hstack((eta_neg, eta_pos))[idx, :]
        km_tick = np.concatenate(
            (
                np.arange(-resolution, -self.mesh.width / 2, -resolution)[::-1],
                np.arange(0, self.mesh.width / 2, resolution),
            )
        )

        ind = (eta_tick >= self.mesh.eta_min) & (eta_tick <= self.mesh.eta_max)
        eta_tick = eta_tick[ind]
        km_tick = km_tick[ind]
        for y, label in zip(eta_tick, km_tick, strict=True):
            self.ax.text(
                xi_tick,
                y,
                str(int(label)),
                horizontalalignment="right",
                verticalalignment="center",
            )

    def text(self, lon, lat, text, ignore_limits=False, **kwargs):
        """Draw text at a geographic position.

        Parameters
        ----------
        lon : float
            The geographic longitude to place the text.
        lat : float
            The geographic latitude to place the text.
        text : str
            The text.
        ignore_limits : bool, optional
            If True, allow text outside the plot limits. The default is False.
        **kwargs : text properties
            Passed to matplotlib.pyplot.text

        Returns
        -------
        Text
            The created Text instance.

        """
        xi, eta = self.mesh.projection.geographic_to_cube(lon, lat)

        if self.mesh.contains(lon, lat) or ignore_limits:
            return self.ax.text(xi, eta, text, **kwargs)
        print('text outside plot limit - set "ignore_limits = True" to override')

    def plot(self, lon, lat, **kwargs):
        """Plot a line using geographic coordinates.

        Parameters
        ----------
        lon : array-like or scalar
            The longitudinal coordinates of the data points.
        lat : array-like or scalar
            The latitudinal coordinates of the data points.
        **kwargs : 2D line properties
            Passed to matplotlib.pyplot.plot.

        Returns
        -------
        list of Line2D
            A list of lines representing the plotted data.

        """
        x, y = self.mesh.projection.geographic_to_cube(lon, lat)
        return self.ax.plot(x, y, **kwargs)

    def scatter(self, lon, lat, **kwargs):
        """Draw points using geographic coordinates.

        Parameters
        ----------
        lon : array-like or scalar
            The longitudinal coordinates of the data points.
        lat : array-like or scalar
            The latitudinal coordinates of the data points.
        **kwargs : 2D line properties
            Passed to matplotlib.pyplot.plot.

        Returns
        -------
        list of Line2D
            A list of lines representing the plotted data.

        """
        x, y = self.mesh.projection.geographic_to_cube(lon, lat)
        return self.ax.scatter(x, y, **kwargs)

    def contour(self, *args, **kwargs):
        """Draw contour lines on the cubed-sphere projection.

        Call signature: contour([X, Y,] Z, **kwargs)

        Parameters
        ----------
        *args : Arrays
            X, Y : array-like, optional
                The lon,lat coordinates of the values in Z.
                X and Y must both be 2D with the same shape as Z
            Z : (M, N) array-like
                The height values over which the contour is drawn.
                Must be of self.mesh.size if X,Y are not provided

        **kwargs : dict
            Passed to matplotlib.pyplot.contour.

        Returns
        -------
        QuadContourSet
            A set of contour lines or filled regions.

        """
        if len(args) == 1:  # Only C provided
            return self.ax.contour(self.mesh.xi, self.mesh.eta, args[0], **kwargs)
        if len(args) == 3:
            x, y = self.mesh.projection.geographic_to_cube(args[0], args[1])
            return self.ax.contour(x, y, args[2], **kwargs)
        raise TypeError("contour accepts either Z or longitude, latitude, Z")

    def contourf(self, *args, **kwargs):
        """Draw filled contours on the cubed-sphere projection.

        Call signature: contourf([X, Y,] Z, **kwargs)

        Parameters
        ----------
        *args : Arrays
            X, Y : array-like, optional
                The lon,lat coordinates of the values in Z.
                X and Y must both be 2D with the same shape as Z
            Z : (M, N) array-like
                The height values over which the filled contour is drawn.
                Must be of self.mesh.size if X,Y are not provided

        **kwargs : dict
            Passed to matplotlib.pyplot.contourf.

        Returns
        -------
        QuadContourSet
            A set of contour lines or filled regions.

        """
        if len(args) == 1:  # Only C provided
            return self.ax.contourf(self.mesh.xi, self.mesh.eta, args[0], **kwargs)
        if len(args) == 3:
            x, y = self.mesh.projection.geographic_to_cube(args[0], args[1])
            return self.ax.contourf(x, y, args[2], **kwargs)
        raise TypeError("contourf accepts either Z or longitude, latitude, Z")

    def pcolormesh(self, *args, **kwargs):
        """Draw a pseudocolor mesh on the cubed-sphere projection.

        Call signature: pcolormesh([X, Y,] Z, **kwargs)

        Parameters
        ----------
        *args : Arrays
            X, Y : array-like, optional
                The lon,lat coordinates of the corners of the values in Z.
                X and Y must both be 2D with shape (M+1,N+1)
            Z : (M, N) array-like
                The values to be plotted.
                Must be of self.mesh.size if X,Y are not provided

        **kwargs : dict
            Passed to matplotlib.pyplot.pcolormesh.

        Returns
        -------
        matplotlib.collections.QuadMesh
            A QuadMesh object.

        """
        if len(args) == 1:  # Only C provided
            return self.ax.pcolormesh(self.mesh.xi_mesh, self.mesh.eta_mesh, args[0], **kwargs)
        if len(args) == 3:
            x, y = self.mesh.projection.geographic_to_cube(args[0], args[1])
            return self.ax.pcolormesh(x, y, args[2], **kwargs)
        raise TypeError("pcolormesh accepts either Z or longitude, latitude, Z")

    def quiver(self, east, north, lon, lat, **kwargs):
        """Draw eastward/northward vectors at geographic positions.

        Parameters
        ----------
        east: array_like or scalar
            Eastward vector components
        north : array_like or scalar
            Northward vector components
        lon : array-like or scalar
            The longitudinal coordinates of the data points.
        lat : array-like or scalar
            The latitudinal coordinates of the data points.
        **kwargs : PolyCollection properties
            Passed to matplotlib.pyplot.quiver

        Returns
        -------
        matplotlib.quiver.Quiver
            A PolyCollection quiver object.

        """
        x, y, xi_component, eta_component = self.mesh.projection.geographic_vector_to_cube(
            east, north, lon, lat
        )

        return self.ax.quiver(x, y, xi_component, eta_component, **kwargs)

    def add_coastlines(self, resolution="110m", **kwargs):
        """Draw Natural Earth coastlines on the cubed-sphere projection.

        Parameters
        ----------
        resolution : str, optional
            Natural Earth resolution identifier.
        **kwargs : 2D line properties
            Passed to matplotlib.pyplot.plot.

        Returns
        -------
        None.

        """
        if "color" not in kwargs:
            kwargs["color"] = "black"

        coastlines = np.load(_DATA_PATH / f"coastlines_{resolution}.npz")
        for key in coastlines:
            lat, lon = coastlines[key]
            xi, eta = self.mesh.projection.geographic_to_cube(lon, lat)
            self.ax.plot(xi, eta, **kwargs)


def circular_mean_degrees(angles, axis=None):
    """Return the circular mean in degrees, ignoring NaNs.

    Parameters
    ----------
    angles : array_like
        Angles in degrees.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute the mean.

    Returns
    -------
    ndarray
        Circular mean values in degrees.

    """
    return np.rad2deg(
        np.arctan2(
            np.nanmean(np.sin(np.deg2rad(angles)), axis=axis),
            np.nanmean(np.cos(np.deg2rad(angles)), axis=axis),
        )
    )


__all__ = ["RegionalCSPlotter", "circular_mean_degrees"]
