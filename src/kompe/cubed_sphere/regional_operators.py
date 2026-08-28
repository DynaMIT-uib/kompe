"""Interpolation and differential operators for a regional CS mesh."""

from functools import cached_property

import numpy as np
from scipy import sparse as scipy_sparse

from kompe.cubed_sphere import cs_coordinates, cs_vectors, finite_differences
from kompe.cubed_sphere.regional_mesh import RegionalCSMesh
from kompe.cubed_sphere.regional_projection import _NORTH_FACE
from kompe.math import as_linear_map, backend_context, get_array_module


def _interpolation_axis(position, first_center, spacing, count):
    """Return neighbouring indices and fractions on one uniform cell-centre axis."""
    xp = get_array_module(position)
    if count == 1:
        index = xp.zeros(position.size, dtype=int)
        return index, index, xp.zeros(position.size)
    coordinate = (position - first_center) / spacing
    lower = xp.clip(xp.floor(coordinate).astype(int), 0, count - 2)
    return lower, lower + 1, coordinate - lower


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
            with a 3-point stencil. With ``stencil_size=2``, a 5-point
            stencil is used, and so on.
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
        coefficients = finite_differences.finite_difference_weights(points, order=1)
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
            coefficients = finite_differences.finite_difference_weights(points, order=1)
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
            coefficients = finite_differences.finite_difference_weights(points, order=1)
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
        """Return the matrix mapping a tangential vector field to divergence.

        The returned N x 2N matrix operates on a 1D array that represents a
        vector field. The array must be of length 2N, where N is the number
        of grid cells. The first N elements are the southward ``theta``
        components and the last N are the eastward ``phi`` components.

        For ``V = (V_theta, V_phi)``, the spherical components are converted
        to contravariant ``(V^xi, V^eta)`` components, then
        ``div(V) = 1/sqrt(g) * partial_a(sqrt(g) V^a)`` is discretized.

        Parameters
        ----------
        stencil_size: int, optional
            Stencil size. Default is 1, in which case derivatives will be calculated
            with a 3-point stencil. With ``stencil_size=2``, a 5-point
            stencil is used, and so on.
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
        with backend_context("numpy"):
            dual_basis_local = cs_vectors._cartesian_to_cube_matrix(
                xi,
                eta,
                radius=grid.radius,
                face=_NORTH_FACE,
            )[:, :2, :].transpose(0, 2, 1)
            metric = cs_coordinates.surface_metric_tensor(xi, eta, radius=grid.radius)
        dual_basis = np.einsum(
            "ij,njk->nik",
            grid.projection.local_to_geographic_matrix,
            dual_basis_local,
        )
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
        """Interpolate a cell-centred scalar field at requested coordinates.

        Bilinear interpolation uses only the
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
        xp = get_array_module(values, lon, lat)
        lon, lat = xp.broadcast_arrays(xp.asarray(lon), xp.asarray(lat))
        scalar_values = xp.asarray(values).reshape(-1)
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
        # Outside values are replaced only for safe array indexing; the final
        # result retains NaN there. This also avoids dynamic boolean slices in
        # JAX-compiled interpolation.
        xi_for_interpolation = xp.where(inside, xi, grid.xi[0, 0])
        eta_for_interpolation = xp.where(inside, eta, grid.eta[0, 0])
        i0, i1, eta_fraction = _interpolation_axis(
            eta_for_interpolation,
            grid.eta[0, 0],
            grid.deta,
            grid.n_eta,
        )
        j0, j1, xi_fraction = _interpolation_axis(
            xi_for_interpolation,
            grid.xi[0, 0],
            grid.dxi,
            grid.n_xi,
        )
        field = scalar_values.reshape(grid.shape)
        interpolated = (
            (1 - eta_fraction) * (1 - xi_fraction) * field[i0, j0]
            + eta_fraction * (1 - xi_fraction) * field[i1, j0]
            + eta_fraction * xi_fraction * field[i1, j1]
            + (1 - eta_fraction) * xi_fraction * field[i0, j1]
        )
        return xp.where(inside, interpolated, xp.nan).reshape(lat.shape)


__all__ = ["RegionalCSOperators"]
