"""Spherical Harmonic Basis Class."""

import math
from collections import OrderedDict

import numpy as np
from scipy.special import assoc_legendre_p_all

from kompe.basis import BasisSubset, SurfaceDifferentialBasis, _owned_readonly_array
from kompe.math import as_linear_map, diagonal_linear_map
from kompe.math.backend import get_array_module, jax_enabled, to_numpy
from kompe.spherical_harmonics.coefficients import (
    SHCoefficientIndices,
    schmidt_quasi_normalization_factors,
)

_EVALUATION_CACHE_VERSION = 1


def _double_factorial(n):
    """Double factorial that correctly handles the n=-1 case."""
    if n < -1:
        # This case is not expected, but defined for completeness.
        raise ValueError("Double factorial is not defined for n < -1 in this context.")
    if n == -1 or n == 0:
        return 1.0
    result = 1.0
    for i in range(n, 0, -2):
        result *= i
    return result


def _normalized_degree_limits(max_degree, max_order):
    """Return validated maximum spherical-harmonic degree and order."""
    if isinstance(max_degree, bool) or not isinstance(max_degree, (int, np.integer)):
        raise TypeError("max_degree must be an integer.")
    if isinstance(max_order, bool) or not isinstance(max_order, (int, np.integer)):
        raise TypeError("max_order must be an integer.")
    max_degree, max_order = int(max_degree), int(max_order)
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative.")
    if max_order < 0 or max_order > max_degree:
        raise ValueError("max_order must be between zero and max_degree.")
    return max_degree, max_order


def _minimum_scalar_degree(min_degree, mean_free):
    """Resolve the minimum scalar degree from gauge-space options."""
    if mean_free is None:
        return 1 if min_degree is None else int(min_degree)
    effective_nmin = 1 if bool(mean_free) else 0
    if min_degree is not None and int(min_degree) != effective_nmin:
        raise ValueError(
            "SHBasis received inconsistent scalar-space options: "
            f"min_degree={min_degree} and mean_free={mean_free}."
        )
    return effective_nmin


class SHBasis(SurfaceDifferentialBasis):
    """Class for representing spherical harmonic bases.

    Uses the Langel (1987) geomagnetism convention.

    This class provides two fully compatible backends for Legendre
    polynomial generation:

    - ``'internal'``:
        A fast, self-contained recurrence relation for both P and dP/dθ.
    - ``'scipy'``:
        Uses the trusted scipy library, with a precise analytical
        scaling factor applied to ensure identical output to the
        ``'internal'`` method.
    """

    _grid_cache_size = 8

    def __init__(
        self,
        max_degree,
        max_order,
        min_degree=None,
        mean_free=None,
        schmidt_quasi_normalized=True,
        legendre_method="internal",
        operator_cache=None,
    ):
        """
        Initialize the SHBasis instance.

        Parameters
        ----------
        max_degree : int
            Maximum degree.
        max_order : int
            Maximum order.
        min_degree : int, optional
            Minimum degree. Defaults to the mean-free scalar space.
        mean_free : bool, optional
            Whether scalar spaces omit the monopole term. If provided,
            it must be consistent with ``min_degree``.
        schmidt_quasi_normalized : bool, optional
            If True, applies Schmidt quasi-normalization factors. By
            default True.
        legendre_method : str, optional
            Method for Legendre function calculation. Can be 'internal'
            (default) or 'scipy'. Both produce identical results.
        operator_cache : object, optional
            Cache implementing ``get_or_create(category, identity, builder)``.
        """
        max_degree, max_order = _normalized_degree_limits(max_degree, max_order)
        if legendre_method not in ["internal", "scipy"]:
            raise ValueError(
                f"Legendre method {legendre_method!r} is not recognized; "
                "use 'internal' or 'scipy'."
            )
        effective_nmin = _minimum_scalar_degree(min_degree, mean_free)
        self.max_degree, self.max_order, self.min_degree, self.legendre_method = (
            max_degree,
            max_order,
            effective_nmin,
            legendre_method,
        )
        self.mean_free = self.min_degree >= 1
        self.operator_cache = operator_cache
        self._related_basis_cache = {}
        self._grid_cache = OrderedDict()
        self._init_coefficient_indices()
        self._init_normalization(schmidt_quasi_normalized)

        if self.legendre_method == "scipy":
            self._compute_scipy_scaling_factors()

        self.kind = "SH"
        self.index_names = ("n", "m")
        self.index_length = self.cnm.n.size + self.snm.n.size
        self.index_arrays = (self.n, self.m)
        self.validate_metadata()

    def __repr__(self):
        """Summarize the harmonic coefficient space."""
        return (
            f"SHBasis(max_degree={self.max_degree}, max_order={self.max_order}, "
            f"min_degree={self.min_degree}, "
            f"schmidt_quasi_normalized={self.schmidt_quasi_normalized}, "
            f"legendre_method={self.legendre_method!r})"
        )

    def _init_coefficient_indices(self):
        """Build cosine/sine coefficient indices and filters."""
        self.index_pairs = tuple(
            (n, m) for n in range(self.max_degree + 1) for m in range(min(self.max_order, n) + 1)
        )
        self._index_map = {pair: index for index, pair in enumerate(self.index_pairs)}
        self.cnm = SHCoefficientIndices(
            pair for pair in self.index_pairs if pair[0] >= self.min_degree
        )
        self.snm = SHCoefficientIndices(
            pair for pair in self.index_pairs if pair[0] >= self.min_degree and pair[1] >= 1
        )

        cnm_pairs = set(self.cnm.index_pairs)
        snm_pairs = set(self.snm.index_pairs)
        self.cnm_filter = _owned_readonly_array(
            [pair in cnm_pairs for pair in self.index_pairs], dtype=bool
        )
        self.snm_filter = _owned_readonly_array(
            [pair in snm_pairs for pair in self.index_pairs], dtype=bool
        )

        self.n = _owned_readonly_array(np.hstack((self.cnm.n.reshape(-1), self.snm.n.reshape(-1))))
        self.m = _owned_readonly_array(np.hstack((self.cnm.m.reshape(-1), self.snm.m.reshape(-1))))
        self.cnm.n = _owned_readonly_array(self.cnm.n)
        self.cnm.m = _owned_readonly_array(self.cnm.m)
        self.snm.n = _owned_readonly_array(self.snm.n)
        self.snm.m = _owned_readonly_array(self.snm.m)

    def _init_normalization(self, schmidt_quasi_normalized):
        """Build immutable coefficient normalization factors."""
        self.schmidt_quasi_normalized = schmidt_quasi_normalized
        s_matrix = schmidt_quasi_normalization_factors(self.max_degree, self.max_order)
        self._schmidt_quasi_factors = _owned_readonly_array(
            [s_matrix[n, m] for n, m in self.index_pairs]
        )
        factors = (
            self._schmidt_quasi_factors
            if self.schmidt_quasi_normalized
            else np.ones(len(self.index_pairs))
        )
        self.schmidt_factors = _owned_readonly_array(factors)

    @property
    def coefficient_space_signature(self):
        """Return a signature for SH coefficient compatibility."""
        return (
            "SH",
            int(self.max_degree),
            int(self.max_order),
            int(self.min_degree),
            bool(self.schmidt_quasi_normalized),
        )

    @staticmethod
    def _grid_cache_key(grid):
        """Return a stable cache key for one grid/array-backend pair."""
        signature = getattr(grid, "signature", None)
        if signature is None:
            return None
        return (signature, bool(jax_enabled()))

    def _evaluation_cache_identity(self, grid, derivative):
        """Return exact identity for one persisted SH evaluation."""
        grid_signature = getattr(grid, "signature", None)
        if grid_signature is None:
            return None
        return {
            "algorithm": "sh_scalar_evaluation",
            "algorithm_version": _EVALUATION_CACHE_VERSION,
            "basis": self.signature,
            "grid_coordinates": grid_signature,
            "derivative": "value" if derivative is None else derivative,
        }

    def _grid_cache_entry(self, grid):
        """Return the cache entry for one stable grid."""
        key = self._grid_cache_key(grid)
        if key is None:
            return None
        cache = self._grid_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        entry = {"legendre": {}, "matrices": {}, "operators": {}}
        cache[key] = entry
        if len(cache) > self._grid_cache_size:
            cache.popitem(last=False)
        return entry

    def _cached_grid_matrix(self, grid, key, build):
        """Return a cached grid matrix when the grid has a signature."""
        entry = self._grid_cache_entry(grid)
        if entry is None:
            return build(None)
        if key not in entry["matrices"]:
            entry["matrices"][key] = build(entry["legendre"])
        return entry["matrices"][key]

    def _cached_grid_operator(self, grid, key, build):
        """Return a cached grid operator when possible."""
        entry = self._grid_cache_entry(grid)
        if entry is None:
            return build()
        if key not in entry["operators"]:
            entry["operators"][key] = build()
        return entry["operators"][key]

    def _operator_from_matrix(self, matrix, *, input_shape):
        """Return a LinearMap shaped like ``matrix``."""
        output_rank = matrix.ndim - len(input_shape)
        return as_linear_map(
            matrix, input_shape=input_shape, output_shape=matrix.shape[:output_rank]
        )

    def omits_constant_mode(self):
        """Return whether scalar coefficients omit the monopole."""
        return self.mean_free

    def _surface_mode_norm(self):
        """Return the L2 norm of each spherical-harmonic surface mode."""
        coefficient_factors = np.hstack(
            (
                self._schmidt_quasi_factors[self.cnm_filter],
                self._schmidt_quasi_factors[self.snm_filter],
            )
        )
        angular_norm = 1.0 / (2.0 * self.n + 1.0)
        if not self.schmidt_quasi_normalized:
            angular_norm = angular_norm / coefficient_factors**2
        return np.sqrt(angular_norm)

    def scalar_smoothness_weights(self):
        """Return gradient-norm weights for scalar coefficients."""
        return self._surface_mode_norm() * np.sqrt(self.n * (self.n + 1.0))

    def helmholtz_smoothness_weights(self):
        """Return gradient-norm weights for Helmholtz field coefficients."""
        return self._surface_mode_norm() * self.n * (self.n + 1.0)

    def with_mean_free(self, mean_free):
        """Return a cached SH scalar-space basis or view."""
        target_mean_free = bool(mean_free)
        if target_mean_free == self.mean_free:
            return self
        if target_mean_free in self._related_basis_cache:
            return self._related_basis_cache[target_mean_free]

        if not self.mean_free and target_mean_free:
            coefficient_indices = np.flatnonzero(self.n >= 1)
            sibling = BasisSubset(
                self,
                coefficient_indices,
                metadata={
                    "max_degree": self.max_degree,
                    "max_order": self.max_order,
                    "min_degree": 1,
                    "mean_free": True,
                    "legendre_method": self.legendre_method,
                    "schmidt_quasi_normalized": self.schmidt_quasi_normalized,
                },
                coefficient_space_signature=(
                    "SH",
                    int(self.max_degree),
                    int(self.max_order),
                    1,
                    bool(self.schmidt_quasi_normalized),
                ),
                subset_name="mean_free",
            )
        else:
            sibling = SHBasis(
                self.max_degree,
                self.max_order,
                mean_free=target_mean_free,
                schmidt_quasi_normalized=self.schmidt_quasi_normalized,
                legendre_method=self.legendre_method,
                operator_cache=self.operator_cache,
            )
        self._related_basis_cache[target_mean_free] = sibling
        sibling._related_basis_cache[bool(self.mean_free)] = self
        return sibling

    def _compute_scipy_scaling_factors(self):
        """Calculate the analytical scaling factor.

        Such that P_internal = F * P_scipy.
        F(n, m) = (n - m)! / (2n - 1)!!
        """
        factors = np.ones(len(self.index_pairs), dtype=np.float64)
        for i, (n, m) in enumerate(self.index_pairs):
            denominator = _double_factorial(2 * n - 1)
            numerator = math.factorial(n - m)
            factors[i] = numerator / denominator
        self.scipy_scaling_factors = factors

    def _get_legendre_scipy(self, theta, compute_derivative=False):
        """Return Legendre functions from SciPy."""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        diff_order = 1 if compute_derivative else 0
        p_and_dp_all = assoc_legendre_p_all(
            self.max_degree, self.max_order, cos_theta, diff_n=diff_order
        )
        p_all, dp_dz_all = (
            (p_and_dp_all[0], p_and_dp_all[1]) if compute_derivative else (p_and_dp_all[0], None)
        )

        P_std = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        dP_std = np.empty_like(P_std) if compute_derivative else None

        for i, (n, m) in enumerate(self.index_pairs):
            p_values = p_all[n, self.max_order + m].T
            cs_phase = (-1) ** m
            P_std[:, i] = p_values * cs_phase
            if compute_derivative:
                dp_dz_values = dp_dz_all[n, self.max_order + m].T
                dp_dz = dp_dz_values * cs_phase
                dP_std[:, i] = dp_dz * (-sin_theta)

        P_scaled = P_std * self.scipy_scaling_factors
        dP_scaled = dP_std * self.scipy_scaling_factors if compute_derivative else None
        return P_scaled, dP_scaled

    def _build_scalar_evaluation_matrix(self, grid, derivative=None):
        """Evaluate scalar basis or surface derivatives on ``grid``."""

        def build(legendre_cache):
            def evaluate():
                return self._evaluate_on_grid(
                    grid, derivative=derivative, legendre_cache=legendre_cache
                )

            identity = self._evaluation_cache_identity(grid, derivative)
            if self.operator_cache is None or identity is None:
                return evaluate()
            cached = self.operator_cache.get_or_create("sh_evaluation", identity, evaluate)
            return get_array_module().asarray(cached)

        return self._cached_grid_matrix(grid, ("scalar_evaluation", derivative), build)

    def _uncached_scalar_evaluation_matrix(self, grid, derivative=None):
        """Evaluate without the persistent array cache."""
        return get_array_module().asarray(self._evaluate_on_grid(grid, derivative=derivative))

    def scalar_evaluation_matrix(self, grid, derivative=None):
        """Return the cached SH scalar evaluation matrix."""
        return self._build_scalar_evaluation_matrix(grid, derivative=derivative)

    def scalar_evaluation_operator(self, grid, derivative=None):
        """Return the cached SH scalar evaluation operator."""
        return self._cached_grid_operator(
            grid,
            ("scalar_evaluation", derivative),
            lambda: self._operator_from_matrix(
                self.scalar_evaluation_matrix(grid, derivative=derivative),
                input_shape=(self.index_length,),
            ),
        )

    def surface_gradient_matrix(self, grid):
        """Return the cached SH surface-gradient matrix."""
        return self._cached_grid_matrix(
            grid,
            "surface_gradient",
            lambda _legendre_cache: self._build_surface_gradient_matrix(grid),
        )

    def _build_surface_gradient_matrix(self, grid):
        """Build the SH surface-gradient matrix."""
        theta_matrix = self.scalar_evaluation_matrix(grid, derivative="theta")
        phi_matrix = self.scalar_evaluation_matrix(grid, derivative="phi")
        xp = get_array_module(theta_matrix, phi_matrix)
        return xp.stack([xp.asarray(theta_matrix), xp.asarray(phi_matrix)])

    def surface_gradient_operator(self, grid):
        """Return the cached SH surface-gradient operator."""
        return self._cached_grid_operator(
            grid,
            "surface_gradient",
            lambda: self._operator_from_matrix(
                self.surface_gradient_matrix(grid), input_shape=(self.index_length,)
            ),
        )

    def rhat_cross_gradient_matrix(self, grid):
        """Return the cached SH r-hat-cross-gradient matrix."""
        return self._cached_grid_matrix(
            grid,
            "rhat_cross_gradient",
            lambda _legendre_cache: self._build_rhat_cross_gradient_matrix(grid),
        )

    def _build_rhat_cross_gradient_matrix(self, grid):
        """Build the SH r-hat-cross-gradient matrix."""
        gradient = self.surface_gradient_matrix(grid)
        xp = get_array_module(gradient)
        return xp.stack([-gradient[1], gradient[0]])

    def rhat_cross_gradient_operator(self, grid):
        """Return the cached SH r-hat-cross-gradient operator."""
        return self._cached_grid_operator(
            grid,
            "rhat_cross_gradient",
            lambda: self._operator_from_matrix(
                self.rhat_cross_gradient_matrix(grid), input_shape=(self.index_length,)
            ),
        )

    def helmholtz_synthesis_matrix(self, grid):
        """Return the cached SH Helmholtz synthesis tensor."""
        return self._cached_grid_matrix(
            grid,
            "helmholtz_synthesis",
            lambda _legendre_cache: self._build_helmholtz_synthesis_matrix(grid),
        )

    def _build_helmholtz_synthesis_matrix(self, grid):
        """Build the SH Helmholtz synthesis tensor."""
        gradient = self.surface_gradient_matrix(grid)
        xp = get_array_module(gradient)
        rotated_gradient = xp.stack([-gradient[1], gradient[0]])
        return xp.stack([-xp.asarray(gradient), rotated_gradient], axis=2)

    def helmholtz_synthesis_operator(self, grid):
        """Return the cached SH Helmholtz synthesis operator."""
        return self._cached_grid_operator(
            grid,
            "helmholtz_synthesis",
            lambda: SurfaceDifferentialBasis.helmholtz_synthesis_operator(self, grid),
        )

    def _normalized_legendre_values(self, theta, *, derivative_required, cache):
        """Return normalized Legendre values and optional dP/dtheta."""
        cached_P = None if cache is None else cache.get("P_unnormalized")
        cached_dP = None if cache is None else cache.get("dP_unnormalized")
        if self.legendre_method == "internal":
            P_unnormalized = cached_P if cached_P is not None else self.legendre(theta)
            if derivative_required:
                dP_unnormalized = (
                    cached_dP
                    if cached_dP is not None
                    else self.legendre_derivative(theta, P=P_unnormalized)
                )
            else:
                dP_unnormalized = cached_dP
        elif cached_P is not None and (not derivative_required or cached_dP is not None):
            P_unnormalized, dP_unnormalized = cached_P, cached_dP
        else:
            P_unnormalized, dP_unnormalized = self._get_legendre_scipy(
                theta, compute_derivative=derivative_required
            )

        if cache is not None:
            cache["P_unnormalized"] = P_unnormalized
            if dP_unnormalized is not None:
                cache["dP_unnormalized"] = dP_unnormalized
        P = P_unnormalized * self.schmidt_factors
        dP = dP_unnormalized * self.schmidt_factors if dP_unnormalized is not None else None
        return P, dP

    def _phi_derivative_values(self, P, dP, phi, sin_theta_values):
        """Evaluate azimuthal derivatives, including the poles."""
        sin_theta = sin_theta_values.reshape(-1, 1)
        phi_col = phi.reshape(-1, 1)
        is_pole = np.abs(sin_theta) <= 1e-12
        m_c, m_s = self.cnm.m, self.snm.m
        num_Gc = -P[:, self.cnm_filter] * m_c * np.sin(m_c * phi_col)
        Gc = np.divide(num_Gc, sin_theta, out=np.zeros_like(num_Gc), where=~is_pole)
        num_Gs = P[:, self.snm_filter] * m_s * np.cos(m_s * phi_col)
        Gs = np.divide(num_Gs, sin_theta, out=np.zeros_like(num_Gs), where=~is_pole)

        pole_rows = np.flatnonzero(is_pole)
        if pole_rows.size:
            cnm_is_m1 = (self.cnm.m == 1).reshape(-1)
            snm_is_m1 = (self.snm.m == 1).reshape(-1)
            cnm_m1_cols = np.flatnonzero(cnm_is_m1)
            snm_m1_cols = np.flatnonzero(snm_is_m1)
            if cnm_m1_cols.size:
                dP_pole = dP[pole_rows][:, self.cnm_filter][:, cnm_is_m1]
                Gc[np.ix_(pole_rows, cnm_m1_cols)] = -dP_pole * np.sin(phi_col[pole_rows])
            if snm_m1_cols.size:
                dP_pole = dP[pole_rows][:, self.snm_filter][:, snm_is_m1]
                Gs[np.ix_(pole_rows, snm_m1_cols)] = dP_pole * np.cos(phi_col[pole_rows])
        return Gc, Gs

    def _evaluate_on_grid(self, grid, derivative=None, legendre_cache=None):
        """Evaluate scalar basis or surface derivatives on ``grid``."""
        xp = get_array_module(grid.phi, grid.theta)
        phi = np.deg2rad(to_numpy(grid.phi))
        theta = np.deg2rad(to_numpy(grid.theta))
        sin_theta_values = np.sin(theta)
        needs_legendre_derivative = derivative == "theta" or (
            derivative == "phi" and np.any(np.abs(sin_theta_values) <= 1e-12)
        )
        P, dP = self._normalized_legendre_values(
            theta, derivative_required=needs_legendre_derivative, cache=legendre_cache
        )

        if derivative is None:
            Gc = P[:, self.cnm_filter] * np.cos(phi.reshape((-1, 1)) * self.cnm.m)
            Gs = P[:, self.snm_filter] * np.sin(phi.reshape((-1, 1)) * self.snm.m)
        elif derivative == "theta":
            Gc = dP[:, self.cnm_filter] * np.cos(phi.reshape((-1, 1)) * self.cnm.m)
            Gs = dP[:, self.snm_filter] * np.sin(phi.reshape((-1, 1)) * self.snm.m)
        elif derivative == "phi":
            Gc, Gs = self._phi_derivative_values(P, dP, phi, sin_theta_values)
        else:
            raise ValueError(f'Invalid derivative "{derivative}".')

        return xp.asarray(np.hstack((Gc, Gs)))

    def legendre(self, theta):
        """Compute un-normalized Legendre functions."""
        theta = np.asarray(theta, dtype=float)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        P = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        P[:, 0] = 1.0
        for nm in range(1, len(self.index_pairs)):
            n, m = self.index_pairs[nm]
            if n == m:
                P[:, nm] = sin_theta * P[:, self._index_map[(n - 1, m - 1)]]
            else:
                if n > m:
                    P[:, nm] = cos_theta * P[:, self._index_map[(n - 1, m)]]
                if n > m + 1:
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    P[:, nm] -= Knm * P[:, self._index_map[(n - 2, m)]]
        return P

    def legendre_derivative(self, theta, P):
        """Compute d/dθ of Legendre functions."""
        theta = np.asarray(theta, dtype=float)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        dP = np.empty_like(P)
        dP[:, 0] = 0.0
        for nm in range(1, len(self.index_pairs)):
            n, m = self.index_pairs[nm]
            if n == m:
                prev_idx = self._index_map[(n - 1, m - 1)]
                dP[:, nm] = sin_theta * dP[:, prev_idx] + cos_theta * P[:, prev_idx]
            else:
                if n > m:
                    prev_idx = self._index_map[(n - 1, m)]
                    dP[:, nm] = cos_theta * dP[:, prev_idx] - sin_theta * P[:, prev_idx]
                if n > m + 1:
                    prev2_idx = self._index_map[(n - 2, m)]
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    dP[:, nm] -= Knm * dP[:, prev2_idx]
        return dP

    def surface_laplacian_operator(self, r=1.0):
        """Return the diagonal spherical-harmonic surface Laplacian."""
        factors = get_array_module().asarray(-self.n * (self.n + 1) / r**2)
        return diagonal_linear_map(
            factors,
            input_shape=(self.index_length,),
            output_shape=(self.index_length,),
        )
