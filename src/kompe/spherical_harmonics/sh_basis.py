"""Spherical Harmonic Basis Class."""

import math
import warnings
from collections import OrderedDict

import numpy as np
import scipy
from packaging import version

from kompe.core import BasisView, SurfaceOperators, _owned_readonly_array
from kompe.math import as_linear_map
from kompe.math.backend import get_array_module, to_numpy, use_jax
from kompe.spherical_harmonics.helpers import (
    SHIndices,
    schmidt_quasi_normalization_factors,
)

_EVALUATION_CACHE_VERSION = 1

# Conditional Import for SciPy Version Compatibility
# Check the SciPy version to import the correct, available function.
_SCIPY_VERSION = version.parse(scipy.__version__)
if _SCIPY_VERSION >= version.parse("1.15.0"):
    _USE_MODERN_SCIPY = True
    from scipy.special import assoc_legendre_p_all

    # Define lpmn as None for clarity (not used in this path).
    lpmn = None
else:
    _USE_MODERN_SCIPY = False
    from scipy.special import lpmn

    # Define assoc_legendre_p_all as None so the name exists for type
    # hinting/clarity.
    assoc_legendre_p_all = None


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


def _normalized_degree_limits(Nmax, Mmax):
    """Return validated maximum spherical-harmonic degree and order."""
    if isinstance(Nmax, bool) or not isinstance(Nmax, (int, np.integer)):
        raise TypeError("Nmax must be an integer.")
    if isinstance(Mmax, bool) or not isinstance(Mmax, (int, np.integer)):
        raise TypeError("Mmax must be an integer.")
    Nmax, Mmax = int(Nmax), int(Mmax)
    if Nmax < 0:
        raise ValueError("Nmax must be non-negative.")
    if Mmax < 0 or Mmax > Nmax:
        raise ValueError("Mmax must be between zero and Nmax.")
    return Nmax, Mmax


def _minimum_scalar_degree(Nmin, mean_free):
    """Resolve the minimum scalar degree from gauge-space options."""
    if mean_free is None:
        return 1 if Nmin is None else int(Nmin)
    effective_nmin = 1 if bool(mean_free) else 0
    if Nmin is not None and int(Nmin) != effective_nmin:
        raise ValueError(
            "SHBasis received inconsistent scalar-space options: "
            f"Nmin={Nmin} and mean_free={mean_free}."
        )
    return effective_nmin


class SHBasis(SurfaceOperators):
    """Class for representing spherical harmonic bases.

    Uses the Langel (1987) geomagnetism convention.

    This class provides two fully compatible backends for Legendre
    polynomial generation:

    - ``'internal'``:
        A fast, self-contained recurrence relation for both P and dP/dθ.
    - ``'scipy'``:
        Uses the trusted scipy library, with a precise analytical
        scaling factor applied to ensure identical output to the
        ``'internal'`` backend. It automatically selects the best
        available scipy function.
    """

    _grid_cache_size = 8

    def __init__(
        self,
        Nmax,
        Mmax,
        Nmin=None,
        mean_free=None,
        quasi_normalized=True,
        backend="internal",
        operator_cache=None,
    ):
        """
        Initialize the SHBasis instance.

        Parameters
        ----------
        Nmax : int
            Maximum degree.
        Mmax : int
            Maximum order.
        Nmin : int, optional
            Minimum degree. Defaults to the mean-free scalar space.
        mean_free : bool, optional
            Whether scalar spaces omit the monopole term. If provided,
            it must be consistent with ``Nmin``.
        quasi_normalized : bool, optional
            If True, applies Schmidt quasi-normalization factors. By
            default True.
        backend : str, optional
            Backend for Legendre function calculation. Can be 'internal'
            (default) or 'scipy'. Both produce identical results.
        operator_cache : object, optional
            Cache implementing ``get_or_create(category, identity, builder)``.
        """
        Nmax, Mmax = _normalized_degree_limits(Nmax, Mmax)
        if backend not in ["internal", "scipy"]:
            raise ValueError(f"Backend '{backend}' not recognized. Use 'internal' or 'scipy'.")
        effective_nmin = _minimum_scalar_degree(Nmin, mean_free)
        self.Nmax, self.Mmax, self.Nmin, self.backend = Nmax, Mmax, effective_nmin, backend
        self.mean_free = self.Nmin >= 1
        self.operator_cache = operator_cache
        self._related_basis_cache = {}
        self._grid_cache = OrderedDict()
        self._init_coefficient_indices()
        self._init_normalization(quasi_normalized)

        # Use the flag set during the conditional import.
        self._use_modern_scipy = _USE_MODERN_SCIPY

        if self.backend == "scipy":
            self._compute_scipy_scaling_factors()

            if not self._use_modern_scipy:
                warnings.warn(
                    f"Your SciPy version ({scipy.__version__}) is older than 1.15.0. Falling "
                    "back to the deprecated 'lpmn' function. Please consider upgrading SciPy.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self.kind = "SH"
        self.index_names = ("n", "m")
        self.index_length = len(self.cnm.index_pairs) + len(self.snm.index_pairs)
        self.index_arrays = (self.n, self.m)
        self.validate_metadata()

    def _init_coefficient_indices(self):
        """Build cosine/sine coefficient indices and filters."""
        all_indices = SHIndices(self.Nmax, self.Mmax)
        self.index_pairs = tuple(all_indices.index_pairs)

        self.cnm = SHIndices(self.Nmax, self.Mmax)
        self.cnm.index_pairs = tuple(pair for pair in self.index_pairs if pair[0] >= self.Nmin)
        self.cnm.make_arrays()
        self.snm = SHIndices(self.Nmax, self.Mmax)
        self.snm.index_pairs = tuple(
            pair for pair in self.index_pairs if pair[0] >= self.Nmin and pair[1] >= 1
        )
        self.snm.make_arrays()

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

    def _init_normalization(self, quasi_normalized):
        """Build immutable coefficient normalization factors."""
        self.is_normalized = quasi_normalized
        if self.is_normalized:
            s_matrix = schmidt_quasi_normalization_factors(self.Nmax, self.Mmax)
            self.schmidt_factors = _owned_readonly_array(
                [s_matrix[n, m] for n, m in self.index_pairs]
            )
        else:
            self.schmidt_factors = _owned_readonly_array(np.ones(len(self.index_pairs)))

    @property
    def coefficient_space_signature(self):
        """Return a signature for SH coefficient compatibility."""
        return ("SH", int(self.Nmax), int(self.Mmax), int(self.Nmin), bool(self.is_normalized))

    @property
    def kind(self):
        """Short identifier for the basis."""
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value

    @property
    def index_names(self):
        """Names of indices used in the basis."""
        return self._index_names

    @index_names.setter
    def index_names(self, value):
        self._index_names = tuple(value)

    @property
    def index_length(self):
        """Total number of basis functions."""
        return self._index_length

    @index_length.setter
    def index_length(self, value):
        self._index_length = value

    @property
    def index_arrays(self):
        """Arrays of indices used in the basis."""
        return self._index_arrays

    @index_arrays.setter
    def index_arrays(self, value):
        self._index_arrays = value

    @staticmethod
    def _grid_cache_key(grid):
        """Return a stable cache key for one grid/backend pair."""
        signature = getattr(grid, "exact_coordinate_signature", None)
        if signature is None:
            signature = getattr(grid, "signature", None)
        if signature is None:
            return None
        return (signature, bool(use_jax()))

    def _evaluation_cache_identity(self, grid, derivative):
        """Return exact identity for one persisted SH evaluation."""
        coordinates = getattr(grid, "exact_coordinate_signature", None)
        if coordinates is None:
            return None
        return {
            "algorithm": "sh_scalar_evaluation",
            "algorithm_version": _EVALUATION_CACHE_VERSION,
            "basis": self.signature,
            "grid_coordinates": coordinates,
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

    def scalar_fields_are_mean_free_by_construction(self):
        """Return whether scalar coefficients omit the monopole."""
        return self.mean_free

    def scalar_index_length(self, mean_free=None):
        """Return scalar coefficient count."""
        return int(self.scalar_degrees(mean_free=mean_free).size)

    def scalar_degrees(self, mean_free=None):
        """Return harmonic degrees for the requested scalar space."""
        target_mean_free = self.mean_free if mean_free is None else bool(mean_free)
        if target_mean_free == self.mean_free:
            return self.n
        if target_mean_free:
            return self.n[1:]
        return np.concatenate([np.array([0], dtype=self.n.dtype), self.n])

    def scalar_orders(self, mean_free=None):
        """Return harmonic orders for the requested scalar space."""
        target_mean_free = self.mean_free if mean_free is None else bool(mean_free)
        if target_mean_free == self.mean_free:
            return self.m
        if target_mean_free:
            return self.m[1:]
        return np.concatenate([np.array([0], dtype=self.m.dtype), self.m])

    def scalar_index_arrays(self, mean_free=None):
        """Return ``(n, m)`` arrays for the requested scalar space."""
        return self.scalar_degrees(mean_free=mean_free), self.scalar_orders(mean_free=mean_free)

    def with_mean_free(self, mean_free):
        """Return a cached SH scalar-space basis or view."""
        target_mean_free = bool(mean_free)
        if target_mean_free == self.mean_free:
            return self
        if target_mean_free in self._related_basis_cache:
            return self._related_basis_cache[target_mean_free]

        if not self.mean_free and target_mean_free:
            coefficient_indices = np.flatnonzero(self.n >= 1)
            sibling = BasisView(
                self,
                coefficient_indices,
                metadata={
                    "Nmax": self.Nmax,
                    "Mmax": self.Mmax,
                    "Nmin": 1,
                    "mean_free": True,
                    "backend": self.backend,
                    "is_normalized": self.is_normalized,
                },
                coefficient_space_signature=(
                    "SH",
                    int(self.Nmax),
                    int(self.Mmax),
                    1,
                    bool(self.is_normalized),
                ),
                view_name="mean_free",
            )
        else:
            sibling = SHBasis(
                self.Nmax,
                self.Mmax,
                mean_free=target_mean_free,
                quasi_normalized=self.is_normalized,
                backend=self.backend,
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
        """Dispatcher for Scipy Legendre function calculation."""
        if self._use_modern_scipy:
            return self._get_legendre_scipy_modern(theta, compute_derivative)
        else:
            return self._get_legendre_scipy_legacy(theta, compute_derivative)

    def _get_legendre_scipy_modern(self, theta, compute_derivative=False):
        """Legendre functions via `assoc_legendre_p_all` function."""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        diff_order = 1 if compute_derivative else 0
        p_and_dp_all = assoc_legendre_p_all(self.Nmax, self.Mmax, cos_theta, diff_n=diff_order)
        p_all, dp_dz_all = (
            (p_and_dp_all[0], p_and_dp_all[1]) if compute_derivative else (p_and_dp_all[0], None)
        )

        P_std = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        dP_std = np.empty_like(P_std) if compute_derivative else None

        for i, (n, m) in enumerate(self.index_pairs):
            p_values = p_all[n, self.Mmax + m].T
            cs_phase = (-1) ** m
            P_std[:, i] = p_values * cs_phase
            if compute_derivative:
                dp_dz_values = dp_dz_all[n, self.Mmax + m].T
                dp_dz = dp_dz_values * cs_phase
                dP_std[:, i] = dp_dz * (-sin_theta)

        P_scaled = P_std * self.scipy_scaling_factors
        dP_scaled = dP_std * self.scipy_scaling_factors if compute_derivative else None
        return P_scaled, dP_scaled

    def _get_legendre_scipy_legacy(self, theta, compute_derivative=False):
        """Legendre functions via `lpmn` function (SciPy<1.15)."""
        theta = np.atleast_1d(theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        P_std = np.empty((theta.size, len(self.index_pairs)), dtype=np.float64)
        dP_std = np.empty_like(P_std) if compute_derivative else None

        for i, (ct, st) in enumerate(zip(cos_theta, sin_theta, strict=True)):
            p_all, dp_dz_all = lpmn(self.Mmax, self.Nmax, ct)
            for j, (n, m) in enumerate(self.index_pairs):
                cs_phase = (-1) ** m
                P_std[i, j] = p_all[m, n] * cs_phase
                if compute_derivative:
                    dp_dz = dp_dz_all[m, n] * cs_phase
                    dP_std[i, j] = dp_dz * (-st)

        P_scaled = P_std * self.scipy_scaling_factors
        dP_scaled = dP_std * self.scipy_scaling_factors if compute_derivative else None
        return P_scaled, dP_scaled

    def evaluate_on_grid(self, grid, derivative=None):
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

    def evaluate_on_grid_uncached(self, grid, derivative=None):
        """Evaluate without the persistent array cache."""
        return get_array_module().asarray(self._evaluate_on_grid(grid, derivative=derivative))

    def get_scalar_evaluation_matrix(self, grid, derivative=None):
        """Return the cached SH scalar evaluation matrix."""
        return self.evaluate_on_grid(grid, derivative=derivative)

    def get_scalar_evaluation_operator(self, grid, derivative=None):
        """Return the cached SH scalar evaluation operator."""
        return self._cached_grid_operator(
            grid,
            ("scalar_evaluation", derivative),
            lambda: self._operator_from_matrix(
                self.get_scalar_evaluation_matrix(grid, derivative=derivative),
                input_shape=(self.index_length,),
            ),
        )

    def get_surface_gradient_matrix(self, grid):
        """Return the cached SH surface-gradient matrix."""
        return self._cached_grid_matrix(
            grid,
            "surface_gradient",
            lambda _legendre_cache: self._build_surface_gradient_matrix(grid),
        )

    def _build_surface_gradient_matrix(self, grid):
        """Build the SH surface-gradient matrix."""
        theta_matrix = self.evaluate_on_grid(grid, derivative="theta")
        phi_matrix = self.evaluate_on_grid(grid, derivative="phi")
        xp = get_array_module(theta_matrix, phi_matrix)
        return xp.stack([xp.asarray(theta_matrix), xp.asarray(phi_matrix)])

    def get_surface_gradient_operator(self, grid):
        """Return the cached SH surface-gradient operator."""
        return self._cached_grid_operator(
            grid,
            "surface_gradient",
            lambda: self._operator_from_matrix(
                self.get_surface_gradient_matrix(grid), input_shape=(self.index_length,)
            ),
        )

    def get_rhat_cross_gradient_matrix(self, grid):
        """Return the cached SH r-hat-cross-gradient matrix."""
        return self._cached_grid_matrix(
            grid,
            "rhat_cross_gradient",
            lambda _legendre_cache: self._build_rhat_cross_gradient_matrix(grid),
        )

    def _build_rhat_cross_gradient_matrix(self, grid):
        """Build the SH r-hat-cross-gradient matrix."""
        gradient = self.get_surface_gradient_matrix(grid)
        xp = get_array_module(gradient)
        return xp.stack([-gradient[1], gradient[0]])

    def get_rhat_cross_gradient_operator(self, grid):
        """Return the cached SH r-hat-cross-gradient operator."""
        return self._cached_grid_operator(
            grid,
            "rhat_cross_gradient",
            lambda: self._operator_from_matrix(
                self.get_rhat_cross_gradient_matrix(grid), input_shape=(self.index_length,)
            ),
        )

    def get_helmholtz_synthesis_matrix(self, grid):
        """Return the cached SH Helmholtz synthesis tensor."""
        return self._cached_grid_matrix(
            grid,
            "helmholtz_synthesis",
            lambda _legendre_cache: self._build_helmholtz_synthesis_matrix(grid),
        )

    def _build_helmholtz_synthesis_matrix(self, grid):
        """Build the SH Helmholtz synthesis tensor."""
        gradient = self.get_surface_gradient_matrix(grid)
        xp = get_array_module(gradient)
        rotated_gradient = xp.stack([-gradient[1], gradient[0]])
        return xp.stack([-xp.asarray(gradient), rotated_gradient], axis=2)

    def get_helmholtz_synthesis_operator(self, grid):
        """Return the cached SH Helmholtz synthesis operator."""
        return self._cached_grid_operator(
            grid,
            "helmholtz_synthesis",
            lambda: SurfaceOperators.get_helmholtz_synthesis_operator(self, grid),
        )

    def _normalized_legendre_values(self, theta, *, derivative_required, cache):
        """Return normalized Legendre values and optional dP/dtheta."""
        cached_P = None if cache is None else cache.get("P_unnormalized")
        cached_dP = None if cache is None else cache.get("dP_unnormalized")
        if self.backend == "internal":
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
        index_map = {pair: i for i, pair in enumerate(self.index_pairs)}
        for nm in range(1, len(self.index_pairs)):
            n, m = self.index_pairs[nm]
            if n == m:
                P[:, nm] = sin_theta * P[:, index_map[(n - 1, m - 1)]]
            else:
                if n > m:
                    P[:, nm] = cos_theta * P[:, index_map[(n - 1, m)]]
                if n > m + 1:
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    P[:, nm] -= Knm * P[:, index_map[(n - 2, m)]]
        return P

    def legendre_derivative(self, theta, P):
        """Compute d/dθ of Legendre functions."""
        theta = np.asarray(theta, dtype=float)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        dP = np.empty_like(P)
        dP[:, 0] = 0.0
        index_map = {pair: i for i, pair in enumerate(self.index_pairs)}
        for nm in range(1, len(self.index_pairs)):
            n, m = self.index_pairs[nm]
            if n == m:
                prev_idx = index_map[(n - 1, m - 1)]
                dP[:, nm] = sin_theta * dP[:, prev_idx] + cos_theta * P[:, prev_idx]
            else:
                if n > m:
                    prev_idx = index_map[(n - 1, m)]
                    dP[:, nm] = cos_theta * dP[:, prev_idx] - sin_theta * P[:, prev_idx]
                if n > m + 1:
                    prev2_idx = index_map[(n - 2, m)]
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    dP[:, nm] -= Knm * dP[:, prev2_idx]
        return dP

    def laplacian(self, r=1.0):
        """Factor to apply the spherical harmonic Laplacian operator."""
        return get_array_module().asarray(-self.n * (self.n + 1) / r**2)
