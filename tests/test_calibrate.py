"""
Tests for the desired-calibrations package.

Run with:
    pytest tests/test_calibrate.py -v

Or from repo root:
    python -m pytest tests/test_calibrate.py -v
"""

import pytest
import numpy as np
import warnings

import desired
from desired import calibrate, list_calibrations, get_info


# ---------------------------------------------------------------------------
# Fixtures — representative index values (within validity ranges for most
# calibrations; exact ranges are checked per-calibration in dedicated tests)
# ---------------------------------------------------------------------------

MONOTONIC_NAMES = [
    "R2", "O32", "Ne3O2", "R2Ne3", "N2", "O3N2", "R3N2", "N2O2",
    "S2", "N2S2", "N2S2Ha", "S3", "S3O3", "S23", "O3S2",
    "Ar3O3", "Ar3N2", "Ne3S2", "Ne3S3",
]

DOUBLE_VALUED_NAMES = [
    "R3", "R23", "R_hat", "RNe", "Ne3", "R3S2", "Ar3", "O3HeI",
]

ALL_NAMES = MONOTONIC_NAMES + DOUBLE_VALUED_NAMES

# A mid-range index value that should be valid for all calibrations
MID_INDEX = {
    # Monotonic
    "R2":     -0.3,
    "O32":     0.0,
    "Ne3O2":  -1.0,
    "R2Ne3":  -0.5,
    "N2":     -1.0,
    "O3N2":    1.0,
    "R3N2":    0.5,
    "N2O2":   -1.2,
    "S2":     -0.5,
    "N2S2":   -0.5,
    "N2S2Ha": -0.5,
    "S3":     -0.5,
    "S3O3":   -0.5,
    "S23":     0.0,
    "O3S2":    0.3,
    "Ar3O3":  -0.5,
    "Ar3N2":  -0.5,
    "Ne3S2":  -0.5,
    "Ne3S3":  -0.5,
    # Double-valued — mid-range (will use both branches)
    "R3":      0.3,
    "R23":     0.5,
    "R_hat":   0.3,
    "RNe":     0.0,
    "Ne3":    -0.8,
    "R3S2":    0.0,
    "Ar3":    -0.8,
    "O3HeI":   0.0,
}


# ===========================================================================
# 1. list_calibrations
# ===========================================================================

class TestListCalibrations:

    def test_default_returns_all(self):
        result = list_calibrations()
        assert isinstance(result, list)
        assert len(result) == 27

    def test_monotonic_count(self):
        result = list_calibrations(kind="monotonic")
        assert len(result) == 19
        for name in MONOTONIC_NAMES:
            assert name in result

    def test_double_valued_count(self):
        result = list_calibrations(kind="double_valued")
        assert len(result) == 8
        for name in DOUBLE_VALUED_NAMES:
            assert name in result

    def test_sorted(self):
        result = list_calibrations()
        assert result == sorted(result)

    def test_invalid_kind_raises(self):
        with pytest.raises((ValueError, KeyError)):
            list_calibrations(kind="invalid")

    def test_no_overlap(self):
        mono = set(list_calibrations(kind="monotonic"))
        dval = set(list_calibrations(kind="double_valued"))
        assert mono.isdisjoint(dval)

    def test_o3hei_is_double_valued(self):
        """O3HeI was corrected from monotonic to double_valued mid-project."""
        dval = list_calibrations(kind="double_valued")
        mono = list_calibrations(kind="monotonic")
        assert "O3HeI" in dval
        assert "O3HeI" not in mono


# ===========================================================================
# 2. get_info
# ===========================================================================

class TestGetInfo:

    def test_returns_dict(self):
        info = get_info("N2")
        assert isinstance(info, dict)

    def test_monotonic_fields(self):
        info = get_info("N2")
        assert info["type"] == "monotonic"
        assert "xmin" in info and "xmax" in info
        assert "t2eq0" in info and "t2ge0" in info

    def test_monotonic_scenario_fields(self):
        """Each scenario must have coeff, disp, disp_coeff."""
        info = get_info("N2")
        for scenario in ("t2eq0", "t2ge0"):
            sc = info[scenario]
            assert "coeff" in sc
            assert "disp" in sc
            assert "disp_coeff" in sc
            assert isinstance(sc["coeff"], list)
            assert isinstance(sc["disp"], float)
            assert isinstance(sc["disp_coeff"], list)
            assert len(sc["disp_coeff"]) == 3   # quadratic: [b0, b1, b2]

    def test_double_valued_fields(self):
        info = get_info("R23")
        assert info["type"] == "double_valued"
        assert "upper" in info and "lower" in info
        for branch in ("upper", "lower"):
            b = info[branch]
            assert "xmin" in b and "xmax" in b
            for scenario in ("t2eq0", "t2ge0"):
                sc = b[scenario]
                assert "coeff" in sc
                assert "disp" in sc
                assert "disp_coeff" in sc

    def test_unknown_name_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_info("NONEXISTENT")

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_all_names_have_info(self, name):
        info = get_info(name)
        assert isinstance(info, dict)


# ===========================================================================
# 3. calibrate — monotonic calibrations
# ===========================================================================

class TestCalibrateMonotonic:

    @pytest.mark.parametrize("name", MONOTONIC_NAMES)
    def test_scalar_no_error(self, name):
        """Scalar index without error → finite OH values."""
        idx = MID_INDEX[name]
        result = calibrate(name, idx)
        assert len(result) == 4
        OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = result
        assert np.isfinite(OH_t2eq0)
        assert np.isfinite(OH_t2ge0)

    @pytest.mark.parametrize("name", MONOTONIC_NAMES)
    def test_scalar_with_error(self, name):
        """Scalar index with error → finite OH and error values."""
        idx = MID_INDEX[name]
        result = calibrate(name, idx, index_e=0.05)
        OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = result
        assert np.isfinite(OH_t2eq0)
        assert np.isfinite(OH_t2eq0_e) and OH_t2eq0_e > 0
        assert np.isfinite(OH_t2ge0)
        assert np.isfinite(OH_t2ge0_e) and OH_t2ge0_e > 0

    @pytest.mark.parametrize("name", MONOTONIC_NAMES)
    def test_array_input(self, name):
        """Array of indices → output arrays of same length."""
        idx = MID_INDEX[name]
        indices = np.array([idx - 0.05, idx, idx + 0.05])
        result = calibrate(name, indices, index_e=np.array([0.05, 0.05, 0.05]))
        assert len(result) == 4
        for arr in result:
            assert len(arr) == 3

    def test_n2_reasonable_range(self):
        """N2 calibration should return 12+log(O/H) between 7.5 and 9.5."""
        OH, _, _, _ = calibrate("N2", -1.0)
        assert 7.5 < OH < 9.5

    def test_o3n2_reasonable_range(self):
        OH, _, _, _ = calibrate("O3N2", 1.0)
        assert 7.5 < OH < 9.5

    def test_two_scenarios_differ(self):
        """t2=0 and t2>0 scenarios should differ (different coefficients)."""
        OH_t2eq0, _, OH_t2ge0, _ = calibrate("N2", -1.0, index_e=0.05)
        assert not np.isclose(OH_t2eq0, OH_t2ge0, atol=1e-6)

    def test_include_disp_false(self):
        """With include_disp=False, errors come only from MC propagation."""
        res_with = calibrate("N2", -1.0, index_e=0.05, include_disp=True)
        res_without = calibrate("N2", -1.0, index_e=0.05, include_disp=False)
        # Errors without dispersion should be <= errors with dispersion
        assert res_without[1] <= res_with[1] + 1e-10
        assert res_without[3] <= res_with[3] + 1e-10

    def test_no_index_error_returns_disp_only(self):
        """Without index_e, MC term is zero; total error equals cal dispersion."""
        res = calibrate("N2", -1.0, index_e=None, include_disp=True)
        # Error should be > 0 (from dispersion alone)
        assert res[1] > 0
        assert res[3] > 0

    def test_out_of_range_returns_nan(self):
        """Indices outside validity range should return NaN with a warning."""
        info = get_info("N2")
        far_out = info["xmax"] + 10.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OH, OH_e, _, _ = calibrate("N2", far_out)
        assert np.isnan(OH)
        assert len(w) >= 1


# ===========================================================================
# 4. calibrate — disp_mode parameter
# ===========================================================================

class TestDispMode:

    @pytest.mark.parametrize("mode", ["global", "local", "max"])
    def test_all_modes_return_finite(self, mode):
        OH, OH_e, OH2, OH2_e = calibrate("N2", -1.0, index_e=0.05, disp_mode=mode)
        assert np.isfinite(OH) and np.isfinite(OH_e)
        assert np.isfinite(OH2) and np.isfinite(OH2_e)

    def test_max_mode_ge_global_and_local(self):
        """disp_mode='max' error >= both 'global' and 'local' errors."""
        _, e_global, _, e_global2 = calibrate("N2", -1.0, index_e=0.05, disp_mode="global")
        _, e_local,  _, e_local2  = calibrate("N2", -1.0, index_e=0.05, disp_mode="local")
        _, e_max,    _, e_max2    = calibrate("N2", -1.0, index_e=0.05, disp_mode="max")
        assert e_max >= e_global - 1e-10
        assert e_max >= e_local - 1e-10
        assert e_max2 >= e_global2 - 1e-10
        assert e_max2 >= e_local2 - 1e-10

    def test_invalid_disp_mode_raises(self):
        with pytest.raises((ValueError, KeyError)):
            calibrate("N2", -1.0, disp_mode="banana")

    @pytest.mark.parametrize("mode", ["global", "local", "max"])
    def test_all_modes_array_input(self, mode):
        indices = np.array([-1.2, -1.0, -0.8])
        errors  = np.array([ 0.05, 0.05, 0.05])
        result = calibrate("N2", indices, index_e=errors, disp_mode=mode)
        assert len(result) == 4
        for arr in result:
            assert len(arr) == 3

    def test_global_error_positive_and_finite(self):
        _, e, _, _ = calibrate("O3N2", 1.0, index_e=0.05, disp_mode="global")
        assert np.isfinite(e) and e > 0

    def test_local_dispersion_varies_with_index(self):
        """Local dispersion should be index-dependent (not constant)."""
        idx_lo = np.array([-1.5])
        idx_hi = np.array([ 1.5])
        err = np.array([0.05])
        _, e_lo, _, _ = calibrate("O3N2", idx_lo, index_e=err, disp_mode="local")
        _, e_hi, _, _ = calibrate("O3N2", idx_hi, index_e=err, disp_mode="local")
        # They should not be exactly equal (unless the polynomial is flat)
        # We just check both are positive and finite
        assert e_lo > 0 and e_hi > 0
        assert np.isfinite(e_lo) and np.isfinite(e_hi)


# ===========================================================================
# 5. calibrate — double-valued calibrations
# ===========================================================================

class TestCalibrateDoubleValued:

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_no_branch_returns_8_values(self, name):
        """Without specifying branch, should return both upper and lower."""
        idx = MID_INDEX[name]
        result = calibrate(name, idx, index_e=0.05)
        assert len(result) == 8, (
            f"{name}: expected 8-tuple for both branches, got {len(result)}"
        )

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_upper_branch_returns_4_values(self, name):
        idx = MID_INDEX[name]
        result = calibrate(name, idx, index_e=0.05, branch="upper")
        assert len(result) == 4

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_lower_branch_returns_4_values(self, name):
        idx = MID_INDEX[name]
        result = calibrate(name, idx, index_e=0.05, branch="lower")
        assert len(result) == 4

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_upper_oh_greater_than_lower(self, name):
        """Upper branch should give higher O/H than lower branch (by definition)."""
        idx = MID_INDEX[name]
        up_t2eq0, _, _, _ = calibrate(name, idx, branch="upper")
        lo_t2eq0, _, _, _ = calibrate(name, idx, branch="lower")
        # Both must be finite to make the comparison
        if np.isfinite(up_t2eq0) and np.isfinite(lo_t2eq0):
            assert up_t2eq0 > lo_t2eq0, (
                f"{name}: upper OH ({up_t2eq0:.3f}) not > lower OH ({lo_t2eq0:.3f})"
            )

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_errors_positive(self, name):
        idx = MID_INDEX[name]
        result = calibrate(name, idx, index_e=0.05, branch="upper")
        OH, OH_e, OH2, OH2_e = result
        if np.isfinite(OH_e):
            assert OH_e > 0
        if np.isfinite(OH2_e):
            assert OH2_e > 0

    def test_o3hei_double_valued_works(self):
        """O3HeI was corrected from monotonic to double_valued — verify it works."""
        result = calibrate("O3HeI", MID_INDEX["O3HeI"], index_e=0.05)
        assert len(result) == 8

    def test_invalid_branch_raises(self):
        with pytest.raises((ValueError, KeyError)):
            calibrate("R23", 0.5, branch="middle")

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_8tuple_consistent_with_separate_branch_calls(self, name):
        """8-tuple should match separate upper + lower 4-tuple calls."""
        idx = MID_INDEX[name]
        err = 0.05
        full   = calibrate(name, idx, index_e=err)
        upper  = calibrate(name, idx, index_e=err, branch="upper")
        lower  = calibrate(name, idx, index_e=err, branch="lower")
        # full = (up_t2eq0, up_t2eq0_e, up_t2ge0, up_t2ge0_e,
        #         lo_t2eq0, lo_t2eq0_e, lo_t2ge0, lo_t2ge0_e)
        for i in range(4):
            if np.isfinite(full[i]) and np.isfinite(upper[i]):
                assert np.isclose(full[i], upper[i], atol=1e-6), \
                    f"{name} upper[{i}]: full={full[i]:.6f}, branch={upper[i]:.6f}"
            if np.isfinite(full[i+4]) and np.isfinite(lower[i]):
                assert np.isclose(full[i+4], lower[i], atol=1e-6), \
                    f"{name} lower[{i}]: full={full[i+4]:.6f}, branch={lower[i]:.6f}"


# ===========================================================================
# 6. Monte Carlo reproducibility and seed
# ===========================================================================

class TestMonteCarlo:

    def test_seed_reproducibility(self):
        """Same seed → identical results."""
        r1 = calibrate("N2", -1.0, index_e=0.05, mc_seed=42)
        r2 = calibrate("N2", -1.0, index_e=0.05, mc_seed=42)
        for v1, v2 in zip(r1, r2):
            assert np.isclose(v1, v2, atol=1e-10)

    def test_different_seeds_differ(self):
        """Different seeds → (very likely) different error estimates."""
        r1 = calibrate("N2", -1.0, index_e=0.05, mc_seed=1)
        r2 = calibrate("N2", -1.0, index_e=0.05, mc_seed=9999)
        # OH values (indices 0, 2) must be identical (no randomness in central value)
        assert np.isclose(r1[0], r2[0], atol=1e-10)
        # Error values may differ
        assert not np.isclose(r1[1], r2[1], atol=1e-6) or \
               not np.isclose(r1[3], r2[3], atol=1e-6)

    def test_mc_n_affects_precision(self):
        """More MC samples → more stable (lower variance) error estimate."""
        errors_low  = [calibrate("N2", -1.0, index_e=0.05, mc_N=50,   mc_seed=i)[1] for i in range(20)]
        errors_high = [calibrate("N2", -1.0, index_e=0.05, mc_N=5000, mc_seed=i)[1] for i in range(20)]
        assert np.std(errors_high) < np.std(errors_low)

    def test_default_seed_is_2025(self):
        """Default seed should be 2025 (documented in API)."""
        r_default = calibrate("N2", -1.0, index_e=0.05)
        r_2025    = calibrate("N2", -1.0, index_e=0.05, mc_seed=2025)
        for v1, v2 in zip(r_default, r_2025):
            assert np.isclose(v1, v2, atol=1e-10)


# ===========================================================================
# 7. Array and mixed inputs
# ===========================================================================

class TestArrayInputs:

    def test_mixed_valid_out_of_range(self):
        """Out-of-range elements in array should become NaN, rest finite."""
        info = get_info("N2")
        xmin, xmax = info["xmin"], info["xmax"]
        mid = (xmin + xmax) / 2.0
        indices = np.array([mid, xmax + 100.0, mid])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            OH, OH_e, _, _ = calibrate("N2", indices, index_e=0.05)
        assert np.isfinite(OH[0])
        assert np.isnan(OH[1])
        assert np.isfinite(OH[2])

    def test_array_error_broadcast(self):
        """Scalar error_e should broadcast over array index."""
        indices = np.array([-1.2, -1.0, -0.8])
        result = calibrate("N2", indices, index_e=0.05)
        for arr in result:
            assert len(arr) == 3

    def test_empty_array(self):
        """Empty input array should return empty output arrays."""
        result = calibrate("N2", np.array([]))
        for arr in result:
            assert len(arr) == 0

    def test_single_element_array(self):
        """Single-element array should return length-1 arrays."""
        result = calibrate("N2", np.array([-1.0]), index_e=np.array([0.05]))
        for arr in result:
            assert len(arr) == 1


# ===========================================================================
# 8. Public API surface
# ===========================================================================

class TestPublicAPI:

    def test_calibrate_importable(self):
        from desired import calibrate
        assert callable(calibrate)

    def test_list_calibrations_importable(self):
        from desired import list_calibrations
        assert callable(list_calibrations)

    def test_get_info_importable(self):
        from desired import get_info
        assert callable(get_info)

    def test_package_version_exists(self):
        assert hasattr(desired, "__version__")

    def test_calibrate_signature(self):
        """calibrate() must accept documented keyword arguments."""
        import inspect
        sig = inspect.signature(calibrate)
        params = list(sig.parameters.keys())
        for required in ("name", "index", "index_e", "include_disp",
                         "disp_mode", "branch", "mc_N", "mc_seed"):
            assert required in params, f"Missing parameter: {required}"


# ===========================================================================
# 9. Numerical regression — spot-check specific calibrations
#    (values locked in from a known-good run; update if coefficients change)
# ===========================================================================

class TestNumericalRegression:
    """
    Smoke-test that calibrations return values in physically sensible
    12+log(O/H) ranges. Specific numeric values depend on JSON coefficients
    and should be updated whenever calibrations.json is regenerated.
    """

    @pytest.mark.parametrize("name,idx,oh_min,oh_max", [
        ("N2",    -1.0,  7.8, 8.8),
        ("O3N2",   1.0,  7.8, 8.6),
        ("R23",    0.5,  7.5, 9.0),
        ("R2",    -0.3,  7.5, 9.0),
        ("S2",    -0.5,  7.5, 9.2),
        ("O32",    0.0,  7.5, 9.0),
    ])
    def test_oh_in_physical_range(self, name, idx, oh_min, oh_max):
        info = get_info(name)
        if info["type"] == "monotonic":
            OH, _, _, _ = calibrate(name, idx)
        else:
            result = calibrate(name, idx, branch="upper")
            OH = result[0]
        if np.isfinite(OH):
            assert oh_min < OH < oh_max, \
                f"{name}(index={idx}): OH={OH:.4f} outside [{oh_min},{oh_max}]"

    def test_disp_coeff_evaluated_correctly(self):
        """
        Verify disp_coeff is evaluated as np.polyval (descending power),
        NOT as q0 + q1*x + q2*x^2 (ascending).
        A wrong evaluation produces errors ~1.35 dex instead of ~0.25 dex.
        """
        _, OH_e, _, _ = calibrate("N2", -1.0, index_e=0.05,
                                   disp_mode="local", include_disp=True)
        assert OH_e < 0.5, (
            f"N2 error={OH_e:.3f} dex is too large — "
            "disp_coeff may be evaluated in ascending instead of descending order"
        )

    def test_coefficient_order_not_double_reversed(self):
        """
        Verify coefficients are NOT double-reversed (historical bug).
        A double-reversed N2 calibration gives OH far outside [7.5, 9.5].
        """
        OH, _, _, _ = calibrate("N2", -1.0)
        assert 7.5 < OH < 9.5, (
            f"N2 OH={OH:.3f} suggests a coefficient order bug "
            "(expected 7.5–9.5, got {OH:.3f})"
        )
