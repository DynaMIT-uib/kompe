"""Spherical point grids used as evaluation and observation locations."""

from functools import cached_property

import numpy as np

from kompe.math import array_fingerprint, content_fingerprint


class SphericalGrid:
    """A collection of spherical sample points without implied topology.

    ``SphericalGrid`` is not a mesh: it does not require cells, neighbours, or
    boundaries.  Use :class:`kompe.StructuredSurfaceMesh` implementations for
    cell-based discretizations.

    Attributes
    ----------
    lat : ndarray
        Flattened array of latitude values in degrees.
    lon : ndarray
        Flattened array of longitude values in degrees.
    theta : ndarray
        Flattened array of colatitude values in degrees.
    phi : ndarray
        Flattened array of longitude values in degrees (same as lon).
    area_weights : ndarray, optional
        Flattened cell-area weights associated with the grid points.
    size : int
        Total number of grid points.

    Notes
    -----
    All coordinate arrays are automatically broadcast to match shapes
    and flattened for internal storage.
    """

    def __init__(self, lat=None, lon=None, theta=None, phi=None, area_weights=None):
        """Initialize the grid object from coordinate inputs.

        Parameters
        ----------
        lat : array-like, optional
            Geographic latitude coordinates in degrees.
        lon : array-like, optional
            Geographic longitude coordinates in degrees.
        theta : array-like, optional
            Spherical colatitude coordinates in degrees.
        phi : array-like, optional
            Spherical longitude coordinates in degrees.
        area_weights : array-like, optional
            Cell-area weights for weighted surface fits. If provided,
            the flattened shape must match the grid size.

        Raises
        ------
        ValueError
            If neither `lat`/`theta` or `lon`/`phi` coordinates are
            provided.

        Notes
        -----
        Either `lat` or `theta` must be provided, and either `lon` or
        `phi` must be provided.
        """
        if (lat is None) == (theta is None):
            raise ValueError("Provide exactly one of latitude or theta.")
        if (lon is None) == (phi is None):
            raise ValueError("Provide exactly one of longitude or phi.")

        latitude = np.asarray(lat, dtype=float) if lat is not None else 90.0 - np.asarray(theta)
        longitude = (
            np.asarray(lon, dtype=float) if lon is not None else np.asarray(phi, dtype=float)
        )
        latitude, longitude = np.broadcast_arrays(latitude, longitude)

        self.lat = np.array(latitude, dtype=float, copy=True).reshape(-1)
        self.lon = np.array(longitude, dtype=float, copy=True).reshape(-1)
        if not np.all(np.isfinite(self.lat)) or not np.all(np.isfinite(self.lon)):
            raise ValueError("SphericalGrid coordinates must be finite.")
        if np.any(np.abs(self.lat) > 90.0):
            raise ValueError("SphericalGrid latitude must be between -90 and 90 degrees.")
        self.theta = 90.0 - self.lat
        self.phi = self.lon.copy()

        self.size = self.lon.size

        if area_weights is not None:
            self.area_weights = np.array(area_weights, dtype=float, copy=True).reshape(-1)
            if self.area_weights.shape != (self.size,):
                raise ValueError("area_weights must match the flattened grid size.")
            if not np.all(np.isfinite(self.area_weights)) or np.any(self.area_weights < 0.0):
                raise ValueError("area_weights must be finite and non-negative.")

        for array in (self.lat, self.lon, self.theta, self.phi):
            array.setflags(write=False)
        if hasattr(self, "area_weights"):
            self.area_weights.setflags(write=False)

    @cached_property
    def signature(self):
        """Return a stable signature for this grid."""
        coordinates = content_fingerprint(
            {
                "theta": np.asarray(self.theta, dtype="<f4"),
                "phi": np.asarray(self.phi, dtype="<f4"),
            }
        )
        return (type(self).__module__, type(self).__qualname__, coordinates)

    @cached_property
    def exact_coordinate_signature(self):
        """Return exact coordinate identity for persisted operators."""
        return (
            array_fingerprint(self.theta, dtype="<f8"),
            array_fingerprint(self.phi, dtype="<f8"),
        )

    @cached_property
    def analysis_signature(self):
        """Return cache identity for coordinate-weighted analysis."""
        if not hasattr(self, "area_weights"):
            return (self.signature, None)
        return (self.signature, array_fingerprint(self.area_weights, dtype="<f8"))

    def same_as(self, other):
        """Return whether another grid has the same coordinates."""
        if self is other:
            return True
        if not isinstance(other, SphericalGrid):
            return False
        return self.signature == other.signature

    def __eq__(self, other):
        """Compare grids by their coordinate signatures."""
        if not isinstance(other, SphericalGrid):
            return NotImplemented
        return self.same_as(other)

    def __repr__(self):
        """Summarize this point grid for interactive inspection."""
        weights = ", area_weights=True" if hasattr(self, "area_weights") else ""
        if self.size == 0:
            return f"SphericalGrid(size=0{weights})"
        return (
            f"SphericalGrid(size={self.size}, "
            f"lat_range=({self.lat.min():g}, {self.lat.max():g}), "
            f"lon_range=({self.lon.min():g}, {self.lon.max():g}){weights})"
        )


__all__ = ["SphericalGrid"]
