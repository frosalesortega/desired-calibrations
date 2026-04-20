"""
DESIRED Strong-Line Metallicity Calibrations
=============================================

Core calibration module implementing the generic calibration engine
for both monotonic and double-valued strong-line indices.

Reference: Rosales-Ortega et al. (2026a)
Project:   DESIRED (DEep Spectra of Ionised REgions Database)
"""

import json
import warnings
import numpy as np
from pathlib import Path

__all__ = ["calibrate", "list_calibrations", "get_info"]

# ---------------------------------------------------------------------------
# Load calibration data
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"

def _load_calibrations():
    """Load the calibration parameter database."""
    with open(_DATA_DIR / "calibrations.json", "r") as f:
        return json.load(f)

_DB = _load_calibrations()

# Build a flat lookup: name -> entry (merging monotonic + double_valued)
_CALIBRATIONS = {}
for _name, _entry in _DB.get("monotonic", {}).items():
    _CALIBRATIONS[_name] = _entry
for _name, _entry in _DB.get("double_valued", {}).items():
    _CALIBRATIONS[_name] = _entry


# ---------------------------------------------------------------------------
# Monte Carlo error propagation
# ---------------------------------------------------------------------------
def _mc_poly_error(x, x_e, coeffs, N=1000, seed=2025):
    """
    Estimate the error in y = polyval(coeffs, x) due to errors in x,
    using Monte Carlo simulation.

    Parameters
    ----------
    x      : 1-D array, input values
    x_e    : 1-D array, 1-sigma errors in x (same shape as x)
    coeffs : array-like, polynomial coefficients in np.polyval order
             (descending power)
    N      : int, number of MC realisations
    seed   : int, random seed for reproducibility

    Returns
    -------
    y_std  : 1-D array, standard deviation of MC realisations (error)
    y_mean : 1-D array, mean of MC realisations
    """
    rng = np.random.default_rng(seed)
    x_samples = rng.normal(loc=x[:, None], scale=x_e[:, None],
                           size=(len(x), N))
    y_samples = np.polyval(coeffs, x_samples)
    return y_samples.std(axis=1), y_samples.mean(axis=1)


# ---------------------------------------------------------------------------
# Dispersion evaluation
# ---------------------------------------------------------------------------
def _eval_disp(xx, disp_global, disp_coeff, mode):
    """
    Evaluate the calibration dispersion for an array of index values.

    Parameters
    ----------
    xx         : 1-D array, index values (within validity range)
    disp_global: float, global calibration dispersion (sigma_fit)
    disp_coeff : list or None, coefficients for the dispersion profile
                 sigma(x), stored in np.polyval order (same convention
                 as the calibration coefficients)
    mode       : str, one of 'global', 'local', 'max'

    Returns
    -------
    sigma : 1-D array, dispersion at each point
    """
    if mode == "global" or disp_coeff is None or len(disp_coeff) == 0:
        return np.full_like(xx, disp_global, dtype=float)

    # Evaluate sigma(x) using np.polyval (coefficients in descending power)
    sigma_local = np.polyval(disp_coeff, xx)
    # Ensure non-negative
    sigma_local = np.maximum(sigma_local, 0.0)

    if mode == "local":
        return sigma_local
    elif mode == "max":
        # max[sigma_cal, sigma(x)] as recommended in Sec. 4.4
        return np.maximum(disp_global, sigma_local)
    else:
        raise ValueError(
            f"disp_mode must be 'global', 'local', or 'max'; got '{mode}'"
        )


# ---------------------------------------------------------------------------
# Internal evaluation helper
# ---------------------------------------------------------------------------
def _evaluate_branch(x, x_e, coeff, disp_global, disp_coeff,
                     xmin, xmax, include_disp, disp_mode, mc_N, mc_seed):
    """
    Evaluate a single polynomial branch over the valid range.

    Returns (y, y_e) arrays with NaN outside the validity range.
    """
    y   = np.full_like(x, np.nan, dtype=float)
    y_e = np.full_like(x, np.nan, dtype=float)

    if xmin is None or xmax is None or len(coeff) == 0:
        return y, y_e
    mask = (x >= xmin) & (x <= xmax)
    xx = x[mask]
    if xx.size == 0:
        return y, y_e

    # Evaluate polynomial
    y[mask] = np.polyval(coeff, xx)

    # Evaluate dispersion at each point
    sigma = _eval_disp(xx, disp_global, disp_coeff, disp_mode)

    # --- Error estimation ---
    if x_e is not None:
        xx_e = x_e[mask]
        mc_err, _ = _mc_poly_error(xx, xx_e, coeff, N=mc_N, seed=mc_seed)
        if include_disp:
            y_e[mask] = np.sqrt(mc_err**2 + sigma**2)
        else:
            y_e[mask] = mc_err
    else:
        # No index errors provided: return dispersion only
        y_e[mask] = sigma

    return y, y_e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def list_calibrations(kind=None):
    """
    List available calibration names.

    Parameters
    ----------
    kind : str or None
        Filter by type: ``'monotonic'``, ``'double_valued'``, or ``None``
        (all).

    Returns
    -------
    list of str
    """
    if kind is None:
        return sorted(_CALIBRATIONS.keys())
    return sorted(name for name, entry in _CALIBRATIONS.items()
                  if entry["type"] == kind)


def get_info(name):
    """
    Return the full parameter dictionary for a calibration.

    Parameters
    ----------
    name : str
        Calibration name (e.g. ``'N2'``, ``'R3'``).

    Returns
    -------
    dict
    """
    if name not in _CALIBRATIONS:
        raise ValueError(
            f"Unknown calibration '{name}'. "
            f"Available: {list_calibrations()}"
        )
    return _CALIBRATIONS[name]


def calibrate(name, index, index_e=None, *,
              include_disp=True, disp_mode="max", branch=None,
              mc_N=1000, mc_seed=2025):
    """
    Apply a DESIRED strong-line metallicity calibration.

    Parameters
    ----------
    name : str
        Calibration name (e.g. ``'N2'``, ``'O3N2'``, ``'R3'``).
    index : float or array_like
        Observed line-ratio index value(s).
    index_e : float, array_like, or None
        1-sigma error(s) on the index. If ``None``, the returned errors
        correspond to the calibration dispersion only.
    include_disp : bool, default True
        If ``True`` **and** ``index_e`` is provided, the calibration
        dispersion is added in quadrature to the propagated MC errors.
        If ``False``, only the MC-propagated errors are returned.
    disp_mode : str, default 'max'
        How to evaluate the calibration dispersion:

        - ``'global'`` — use the global dispersion sigma_fit (single
          value for the entire calibration range).
        - ``'local'``  — use the index-dependent dispersion
          sigma(x) = q0 + q1*x + q2*x^2 (Eq. 3 in Rosales-Ortega
          et al. 2026a), evaluated at each data point.
        - ``'max'``    — use max[sigma_cal, sigma(x)] at each point
          (recommended; see Sec. 4.4 of Rosales-Ortega et al. 2026a).
    branch : str or None
        For double-valued calibrations only: ``'upper'``, ``'lower'``,
        or ``None``.  If ``None`` **and** the calibration is
        double-valued, **both** branches are returned (see Returns).
    mc_N : int, default 1000
        Number of Monte Carlo realisations for error propagation.
    mc_seed : int, default 2025
        Random seed for MC reproducibility.

    Returns
    -------
    For **monotonic** calibrations (or double-valued with ``branch``
    specified):

        ``(OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e)``

    For **double-valued** calibrations with ``branch=None``:

        ``(OH_upper_t2eq0, OH_upper_t2eq0_e, OH_upper_t2ge0, OH_upper_t2ge0_e,
          OH_lower_t2eq0, OH_lower_t2eq0_e, OH_lower_t2ge0, OH_lower_t2ge0_e)``

    Each element is a float (if scalar input) or a NumPy array.
    Values outside the validity range are ``NaN``.

    Notes
    -----
    The total uncertainty in a strong-line metallicity estimate is:

        sigma_total = sqrt(sigma_MC^2 + sigma_cal^2)

    where sigma_MC is the Monte Carlo propagated error from the index
    measurement uncertainty, and sigma_cal is the calibration dispersion
    controlled by ``disp_mode``. The recommended setting (``disp_mode='max'``)
    uses max[sigma_fit, sigma(x)], ensuring that the uncertainty is never
    underestimated in sparsely sampled regimes.

    Examples
    --------
    >>> from desired import calibrate
    >>> # Monotonic calibration (recommended: default disp_mode='max')
    >>> OH, OH_e, OH_t2, OH_t2e = calibrate('N2', -1.2, index_e=0.05)
    >>> # Use global dispersion only
    >>> OH, OH_e, OH_t2, OH_t2e = calibrate('N2', -1.2, index_e=0.05,
    ...                                      disp_mode='global')
    >>> # Double-valued, specific branch
    >>> OH, OH_e, OH_t2, OH_t2e = calibrate('R3', 0.6, branch='upper')
    """
    # --- Validate name ---
    if name not in _CALIBRATIONS:
        raise ValueError(
            f"Unknown calibration '{name}'. "
            f"Available: {list_calibrations()}"
        )
    cal = _CALIBRATIONS[name]

    # --- Validate disp_mode ---
    if disp_mode not in ("global", "local", "max"):
        raise ValueError(
            f"disp_mode must be 'global', 'local', or 'max'; "
            f"got '{disp_mode}'"
        )

    # --- Prepare input ---
    x = np.asarray(index, dtype=float)
    scalar_input = (x.ndim == 0)
    x = np.atleast_1d(x)

    if index_e is not None:
        x_e = np.atleast_1d(np.asarray(index_e, dtype=float))
        x_e = np.nan_to_num(x_e, nan=0.0)
        if x.shape != x_e.shape:
            raise ValueError("index and index_e must have the same shape")
    else:
        x_e = None

    # --- Helper to format output ---
    def _format(arrays):
        """Round and convert scalars."""
        rounded = [np.round(a, 2) for a in arrays]
        if scalar_input:
            return tuple(float(a.item()) for a in rounded)
        return tuple(rounded)

    # --- Helper to extract disp_coeff from a scenario dict ---
    def _get_disp_coeff(sc):
        """Get disp_coeff list, or None if absent/empty."""
        dc = sc.get("disp_coeff", None)
        if dc is not None and len(dc) > 0:
            return dc
        return None

    # --- Monotonic calibrations ---
    if cal["type"] == "monotonic":
        if branch is not None:
            warnings.warn(
                f"'{name}' is monotonic; 'branch' argument is ignored.",
                UserWarning, stacklevel=2
            )
        t2eq0 = cal["t2eq0"]
        t2ge0 = cal["t2ge0"]

        y0, y0e = _evaluate_branch(
            x, x_e, t2eq0["coeff"], t2eq0["disp"],
            _get_disp_coeff(t2eq0),
            cal["xmin"], cal["xmax"], include_disp, disp_mode, mc_N, mc_seed
        )
        y1, y1e = _evaluate_branch(
            x, x_e, t2ge0["coeff"], t2ge0["disp"],
            _get_disp_coeff(t2ge0),
            cal["xmin"], cal["xmax"], include_disp, disp_mode, mc_N, mc_seed
        )
        return _format([y0, y0e, y1, y1e])

    # --- Double-valued calibrations ---
    if cal["type"] == "double_valued":
        branches_to_eval = []

        if branch is not None:
            branch = branch.lower()
            if branch not in ("upper", "lower"):
                raise ValueError(
                    f"branch must be 'upper', 'lower', or None; got '{branch}'"
                )
            branches_to_eval = [branch]
        else:
            branches_to_eval = ["upper", "lower"]

        all_results = []
        for br in branches_to_eval:
            br_data = cal[br]
            xmin = br_data["xmin"]
            xmax = br_data["xmax"]

            # Warn if all values fall outside validity range
            mask_valid = ((x >= xmin) & (x <= xmax)
                          if (xmin is not None and xmax is not None)
                          else np.zeros_like(x, dtype=bool))
            if not np.any(mask_valid):
                warnings.warn(
                    f"All index values are outside the validity range "
                    f"[{xmin}, {xmax}] for '{name}' {br} branch. "
                    f"Returned values will be NaN.",
                    UserWarning, stacklevel=2
                )

            for scenario in ("t2eq0", "t2ge0"):
                sc = br_data[scenario]
                y, ye = _evaluate_branch(
                    x, x_e, sc["coeff"], sc["disp"],
                    _get_disp_coeff(sc),
                    xmin, xmax, include_disp, disp_mode, mc_N, mc_seed
                )
                all_results.extend([y, ye])

        return _format(all_results)

    raise ValueError(f"Unknown calibration type: {cal['type']}")
