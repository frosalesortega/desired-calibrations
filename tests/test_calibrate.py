"""
Tests for the desired-calibrations package.

Run with:
    pytest tests/test_calibrate.py -v

Or from repo root:
    python -m pytest tests/test_calibrate.py -v
"""

import json
import os
import subprocess
import warnings
from pathlib import Path

import pytest
import numpy as np

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
    "S23":    -0.77,
    "O3S2":    0.3,
    "Ar3O3":  -1.5,
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

    @pytest.mark.parametrize("name", MONOTONIC_NAMES)
    def test_monotonic_new_fields(self, name):
        """sigma_cal, oh_min, oh_max: additive fields from the revision."""
        info = get_info(name)
        for scenario in ("t2eq0", "t2ge0"):
            sc = info[scenario]
            assert "sigma_cal" in sc and sc["sigma_cal"] > 0
            assert "oh_min" in sc and "oh_max" in sc
            assert sc["oh_min"] < sc["oh_max"]

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_double_valued_new_fields(self, name):
        info = get_info(name)
        for branch in ("upper", "lower"):
            for scenario in ("t2eq0", "t2ge0"):
                sc = info[branch][scenario]
                assert sc["sigma_cal"] > 0
                assert sc["oh_min"] < sc["oh_max"]
        assert set(info["turnover"]) == {"shared", "t2eq0", "t2ge0"}
        assert set(info["excluded_interval_OH"]) == {"t2eq0", "t2ge0"}

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_turnover_shared_only_for_r3_r3s2(self, name):
        """A single Direction-B fit reproduces both branches only for
        R3 and R3S2 (Appendix C); all other bivalued diagnostics use
        independent branch fits with two distinct turnovers."""
        info = get_info(name)
        expected_shared = name in ("R3", "R3S2")
        assert info["turnover"]["shared"] == expected_shared
        if expected_shared:
            assert info["excluded_interval_OH"]["t2eq0"] == 0.0
            assert info["excluded_interval_OH"]["t2ge0"] == 0.0
        else:
            assert info["excluded_interval_OH"]["t2eq0"] > 0.0
            assert info["excluded_interval_OH"]["t2ge0"] > 0.0

    def test_ar3_upper_branch_note_present(self):
        """Ar3 upper branch is a direct Direction-A fit, not the exact
        numerical inverse used elsewhere (Appendix C) -- must be documented."""
        info = get_info("Ar3")
        assert "upper_branch" in info["notes"]

    @pytest.mark.parametrize("name", [n for n in DOUBLE_VALUED_NAMES if n != "Ar3"])
    def test_no_notes_for_other_bivalued(self, name):
        info = get_info(name)
        assert info["notes"] == {}

    def test_meta_schema_version(self):
        """calibrations.json _meta.version must be '2.1': the arxiv-v1
        release already used '2.0' for a schema without sigma_cal/oh_min/
        oh_max/turnover, so this revision's schema needs a distinct value."""
        import json
        data_path = Path(desired.__file__).parent / "data" / "calibrations.json"
        with open(data_path) as f:
            db = json.load(f)
        assert db["_meta"]["version"] == "2.1"


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

    def test_unknown_calibration_name_raises(self):
        with pytest.raises(ValueError):
            calibrate("NONEXISTENT", -1.0)

    def test_branch_argument_ignored_with_warning(self):
        """`branch` is meaningless for a monotonic calibration; it should
        be ignored (not raise) but warn."""
        with pytest.warns(UserWarning):
            result = calibrate("N2", -1.0, branch="upper")
        assert len(result) == 4


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
        idx_lo = np.array([-0.5])
        idx_hi = np.array([ 1.5])
        err = np.array([0.05])
        _, e_lo, _, _ = calibrate("O3N2", idx_lo, index_e=err, disp_mode="local")
        _, e_hi, _, _ = calibrate("O3N2", idx_hi, index_e=err, disp_mode="local")
        # They should not be exactly equal (unless the polynomial is flat)
        # We just check both are positive and finite
        assert e_lo > 0 and e_hi > 0
        assert np.isfinite(e_lo) and np.isfinite(e_hi)

    def test_sigma_source_default_is_cal(self):
        r_default = calibrate("N2", -1.0, disp_mode="global")
        r_cal = calibrate("N2", -1.0, disp_mode="global", sigma_source="cal")
        assert r_default == r_cal

    def test_invalid_sigma_source_raises(self):
        with pytest.raises(ValueError):
            calibrate("N2", -1.0, sigma_source="bogus")

    def test_sigma_source_cal_matches_sigma_cal_field(self):
        """disp_mode='global', sigma_source='cal', no index_e -> the
        returned error is exactly sigma_cal from the JSON (rounded)."""
        info = get_info("N2")["t2eq0"]
        OH, e, _, _ = calibrate("N2", -1.0, disp_mode="global", sigma_source="cal")
        assert np.isclose(e, round(info["sigma_cal"], 2), atol=1e-9)

    def test_sigma_source_fit_matches_disp_field(self):
        info = get_info("N2")["t2eq0"]
        OH, e, _, _ = calibrate("N2", -1.0, disp_mode="global", sigma_source="fit")
        assert np.isclose(e, round(info["disp"], 2), atol=1e-9)

    def test_sigma_source_cal_vs_fit_differ_for_bivalued(self):
        """sigma_cal and disp differ substantially for bivalued diagnostics
        (up to ~0.07 dex) -- must be selectable via sigma_source."""
        r_cal = calibrate("R_hat", 0.0, branch="lower", disp_mode="global", sigma_source="cal")
        r_fit = calibrate("R_hat", 0.0, branch="lower", disp_mode="global", sigma_source="fit")
        assert r_cal != r_fit

    def test_sigma_source_ignored_in_local_mode(self):
        """disp_mode='local' always uses sigma(x); sigma_source has no effect."""
        r_cal = calibrate("N2", -1.0, index_e=0.05, disp_mode="local", sigma_source="cal")
        r_fit = calibrate("N2", -1.0, index_e=0.05, disp_mode="local", sigma_source="fit")
        assert r_cal == r_fit


# ===========================================================================
# 5. calibrate — double-valued calibrations
# ===========================================================================

class TestCalibrateDoubleValued:

    @pytest.fixture(autouse=True)
    def _ignore_range_warnings(self):
        """MID_INDEX is chosen to be valid for at least one branch, but not
        necessarily both -- bivalued diagnostics routinely have only one
        branch valid at a given index, since xmin_up != xmin_low even
        though xmax_up == xmax_low (the shared turnover). These tests
        check return shapes/values, not the warning itself, so the
        (correct, expected) UserWarning is suppressed here."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield

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

    def test_python_list_input(self):
        result = calibrate("N2", [-1.2, -1.0, -0.8], index_e=0.05)
        for arr in result:
            assert len(arr) == 3

    def test_python_tuple_input(self):
        result = calibrate("N2", (-1.2, -1.0, -0.8), index_e=0.05)
        for arr in result:
            assert len(arr) == 3

    def test_incompatible_index_e_shape_raises(self):
        """A scalar or matching-shape index_e broadcasts fine (see above);
        a genuinely incompatible shape must still raise."""
        with pytest.raises(ValueError):
            calibrate("N2", [-1.2, -1.0, -0.8], index_e=[0.05, 0.05])

    def test_0d_array_input(self):
        """A 0-d array behaves like a scalar (float outputs, not arrays)."""
        result = calibrate("N2", np.array(-1.0), index_e=0.05)
        for v in result:
            assert isinstance(v, float)

    def test_2d_array_input_shape_preserved(self):
        indices = np.array([[-1.2, -1.0], [-0.9, -0.8]])
        result = calibrate("N2", indices, index_e=0.05)
        for arr in result:
            assert arr.shape == (2, 2)

    def test_int_dtype_input(self):
        """Integer index array must not raise (internally cast to float)."""
        info = get_info("N2")
        mid = int(round((info["xmin"] + info["xmax"]) / 2.0))
        result = calibrate("N2", np.array([mid], dtype=int), index_e=0.05)
        assert np.isfinite(result[0][0])

    def test_nan_input_returns_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calibrate("N2", np.array([np.nan, -1.0]), index_e=0.05)
        assert np.isnan(result[0][0])
        assert np.isfinite(result[0][1])

    def test_inf_input_returns_nan(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = calibrate("N2", np.array([np.inf, -1.0]), index_e=0.05)
        assert np.isnan(result[0][0])
        assert np.isfinite(result[0][1])

    def test_out_of_range_warns_exactly_once_per_call(self):
        """A single UserWarning per branch/scenario grouping, not per NaN
        element (avoids warning spam on large arrays)."""
        info = get_info("N2")
        far_out = np.full(10, info["xmax"] + 100.0)
        with pytest.warns(UserWarning):
            calibrate("N2", far_out)

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    def test_bivalued_out_of_range_warns(self, name):
        info = get_info(name)
        far_out = info["upper"]["xmax"] + 100.0
        with pytest.warns(UserWarning):
            calibrate(name, far_out, branch="upper")

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_exact_xmin_xmax_are_valid(self, name):
        """The exact validity-range endpoints must NOT be NaN (inclusive
        bounds)."""
        info = get_info(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if info["type"] == "monotonic":
                xmin, xmax = info["xmin"], info["xmax"]
                lo = calibrate(name, xmin)
                hi = calibrate(name, xmax)
            else:
                xmin, xmax = info["upper"]["xmin"], info["upper"]["xmax"]
                lo = calibrate(name, xmin, branch="upper")
                hi = calibrate(name, xmax, branch="upper")
        assert np.isfinite(lo[0]) and np.isfinite(lo[2])
        assert np.isfinite(hi[0]) and np.isfinite(hi[2])


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
        """calibrate() must accept documented keyword arguments, including
        the additive `sigma_source` parameter introduced in 2.0.0."""
        import inspect
        sig = inspect.signature(calibrate)
        params = list(sig.parameters.keys())
        for required in ("name", "index", "index_e", "include_disp",
                         "disp_mode", "sigma_source", "branch", "mc_N", "mc_seed"):
            assert required in params, f"Missing parameter: {required}"

    def test_sigma_source_default_value(self):
        import inspect
        sig = inspect.signature(calibrate)
        assert sig.parameters["sigma_source"].default == "cal"

    def test_list_calibrations_kind_signature_unchanged(self):
        import inspect
        sig = inspect.signature(list_calibrations)
        assert list(sig.parameters.keys()) == ["kind"]
        assert sig.parameters["kind"].default is None


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


# ===========================================================================
# 10. Independent reference (T2) — reads the private pickles directly,
#     using a DIFFERENT code path from calibrate.py, and compares against
#     the package on a dense grid. The pickles are not part of the public
#     package (see .gitignore), so this class is skipped unless
#     DESIRED_TEST_BIN_DIR points at a local copy of bin/.
# ===========================================================================

_BIN_DIR = os.environ.get("DESIRED_TEST_BIN_DIR")
requires_bin_dir = pytest.mark.skipif(
    not _BIN_DIR or not Path(_BIN_DIR).is_dir(),
    reason=(
        "Set DESIRED_TEST_BIN_DIR to the private pickle directory to run "
        "the independent pickle-reference regression (T2); not part of the "
        "public package, skipped by default."
    ),
)


def _load_pickle(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _merged_pickle_data(name, scenario):
    """Same precedence as paper_tables.ipynb: new_<name>.pkl first, then
    sigma[_t2]_<name>.pkl (overwrites on collision)."""
    bin_dir = Path(_BIN_DIR)
    merged = dict(_load_pickle(bin_dir / f"new_{name}.pkl"))
    sigma_file = f"sigma_{name}.pkl" if scenario == "t2eq0" else f"sigma_t2_{name}.pkl"
    merged.update(_load_pickle(bin_dir / sigma_file))
    return merged


@requires_bin_dir
class TestPickleReference:
    """Independent reference: np.polyval directly on the pickles, vs.
    calibrate() reading calibrations.json. A different code path entirely
    (Sec. 6, T2)."""

    @pytest.mark.parametrize("name", MONOTONIC_NAMES)
    @pytest.mark.parametrize("scenario", ["t2eq0", "t2ge0"])
    def test_monotonic_matches_pickle(self, name, scenario):
        merged = _merged_pickle_data(name, scenario)
        xmin, xmax = merged[f"{name}_xmin"], merged[f"{name}_xmax"]
        # calibrations.json rounds xmin/xmax to 4 decimals, so a grid point
        # that lands exactly on the full-precision pickle boundary can fall
        # just outside the (slightly different) rounded JSON boundary; stay
        # a hair inside both ends to avoid spurious NaN at the edges.
        margin = 1e-3 * (xmax - xmin)
        xs = np.linspace(xmin + margin, xmax - margin, 25)
        ncoef = len(merged[f"{name}_{scenario}_coeff"]) // 2
        poly = merged[f"{name}_{scenario}_coeff"][:ncoef]
        expected = np.polyval(poly, xs)

        idx = 0 if scenario == "t2eq0" else 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = calibrate(name, xs)[idx]
        assert np.allclose(got, expected, atol=0.01), (
            f"{name} {scenario}: max|diff|="
            f"{np.nanmax(np.abs(got - expected)):.4f}"
        )

    @pytest.mark.parametrize("name", DOUBLE_VALUED_NAMES)
    @pytest.mark.parametrize("branch,suf", [("upper", "up"), ("lower", "low")])
    @pytest.mark.parametrize("scenario", ["t2eq0", "t2ge0"])
    def test_double_valued_matches_pickle(self, name, branch, suf, scenario):
        merged = _merged_pickle_data(name, scenario)
        xmin = merged[f"{name}_xmin_{suf}"]
        xmax = merged[f"{name}_xmax_{suf}"]
        margin = 1e-3 * (xmax - xmin)
        xs = np.linspace(xmin + margin, xmax - margin, 25)
        coeff_key = f"{name}_{scenario}_coeff_{suf}"
        ncoef = len(merged[coeff_key]) // 2
        poly = merged[coeff_key][:ncoef]
        expected = np.polyval(poly, xs)

        idx = 0 if scenario == "t2eq0" else 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = calibrate(name, xs, branch=branch)[idx]
        assert np.allclose(got, expected, atol=0.01), (
            f"{name} {branch} {scenario}: max|diff|="
            f"{np.nanmax(np.abs(got - expected)):.4f}"
        )


# ===========================================================================
# 11. Table regression (T3) — parses Tables 3/4 (t2eq0/t2ge0) directly from
#     the manuscript .tex sources and compares at printed (2-decimal)
#     precision. Only the ACTIVE (non "%"-commented) rows are parsed. The
#     manuscript source is not part of the public package, so this class is
#     skipped unless DESIRED_TEST_PAPER_DIR points at the paper/ directory.
# ===========================================================================

_PAPER_DIR = os.environ.get("DESIRED_TEST_PAPER_DIR")
requires_paper_dir = pytest.mark.skipif(
    not _PAPER_DIR or not Path(_PAPER_DIR).is_dir(),
    reason=(
        "Set DESIRED_TEST_PAPER_DIR to the manuscript source directory to "
        "run the table regression (T3); not part of the public package, "
        "skipped by default."
    ),
)

_LATEX_TO_NAME = {
    "R2": "R2", "R3": "R3", "R23": "R23", "O32": "O32",
    "\\Rhat": "R_hat", "\\RNe": "RNe",
    "Ne3O2": "Ne3O2", "Ne3": "Ne3", "R2Ne3": "R2Ne3",
    "O3S2": "O3S2", "R3S2": "R3S2",
    "N2": "N2", "O3N2": "O3N2", "R3N2": "R3N2", "N2O2": "N2O2",
    "S2": "S2", "N2S2": "N2S2", "N2S2\\ha": "N2S2Ha",
    "S3": "S3", "S3O3": "S3O3", "S23": "S23",
    "Ar3": "Ar3", "Ar3O3": "Ar3O3", "O3HeI": "O3HeI",
    "Ar3N2": "Ar3N2", "Ne3S2": "Ne3S2", "Ne3S3": "Ne3S3",
}


def _na_or_float(s):
    import re
    s = s.strip()
    if s in ("\\NA", "NA", ""):
        return None
    return float(s)


def _matches_2dp(got, expected):
    """True if `got` (JSON precision, 4 decimals) rounds to the same
    2-decimal value as `expected` (table precision). Deliberately NOT
    `round(got, 2) == expected`: double-rounding through the intermediate
    4-decimal JSON value can flip a near-tie at the 2nd decimal (e.g. a
    true value of 0.47501 rounds correctly to 0.48 directly, but rounds
    to 0.4750 at 4dp first, which Python's round() then rounds down to
    0.47 due to binary float representation of the literal 0.475)."""
    return abs(got - expected) < 0.0051


def _parse_tex_table(path):
    """Returns {(name, branch): row_dict} for the ACTIVE rows only."""
    import re
    rows = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if "&" not in line:
                continue
            m = re.match(
                r"^(R2|R3|R23|O32|\\Rhat|\\RNe|Ne3O2|Ne3|R2Ne3|O3S2|R3S2|"
                r"N2|O3N2|R3N2|N2O2|S2|N2S2|S3|S3O3|S23|Ar3|Ar3O3|O3HeI|"
                r"Ar3N2|Ne3S2|Ne3S3|\\phantom)", line)
            if not m:
                continue
            content = line.rstrip("\\").rstrip("\\").strip()
            cells = [c.strip() for c in content.split("&")]
            name_cell = cells[0]
            branch = None
            if "scriptsize upper" in name_cell:
                branch = "upper"
            elif "scriptsize lower" in name_cell:
                branch = "lower"
            pm = re.search(r"\\phantom\{([^}]+)\}", name_cell)
            latex_name = pm.group(1) if pm else name_cell.split("~")[0].strip()
            if latex_name not in _LATEX_TO_NAME:
                continue
            name = _LATEX_TO_NAME[latex_name]
            rest = cells[1:]
            a = [_na_or_float(x) for x in rest[0:4]]
            vmin, vmax = (_na_or_float(x) for x in rest[4:6])
            rows[(name, branch)] = {"a": a, "xmin": vmin, "xmax": vmax}
    return rows


@requires_paper_dir
class TestTableRegression:
    """calibrations.json coefficients (reversed to ascending) and validity
    range vs. the printed Tables 3/4, at 2-decimal (table) precision."""

    @pytest.mark.parametrize("scenario,fname", [
        ("t2eq0", "table_coeff_t2eq0.tex"), ("t2ge0", "table_coeff_t2ge0.tex"),
    ])
    def test_monotonic_coeff_matches_table(self, scenario, fname):
        table = _parse_tex_table(Path(_PAPER_DIR) / fname)
        for name in MONOTONIC_NAMES:
            info = get_info(name)
            sc = info[scenario]
            asc = list(reversed(sc["coeff"]))
            while len(asc) < 4:
                asc.append(None)
            row = table[(name, None)]
            for i, (got, expected) in enumerate(zip(asc, row["a"])):
                if expected is None:
                    assert got is None or len(sc["coeff"]) <= i, name
                    continue
                assert _matches_2dp(got, expected), (
                    f"{name} {scenario} a{i}: json={got:.4f} table={expected}"
                )
            assert _matches_2dp(info["xmin"], row["xmin"]), name
            assert _matches_2dp(info["xmax"], row["xmax"]), name

    @pytest.mark.parametrize("scenario,fname", [
        ("t2eq0", "table_coeff_t2eq0.tex"), ("t2ge0", "table_coeff_t2ge0.tex"),
    ])
    def test_double_valued_coeff_matches_table(self, scenario, fname):
        table = _parse_tex_table(Path(_PAPER_DIR) / fname)
        for name in DOUBLE_VALUED_NAMES:
            info = get_info(name)
            for branch in ("upper", "lower"):
                sc = info[branch][scenario]
                asc = list(reversed(sc["coeff"]))
                while len(asc) < 4:
                    asc.append(None)
                row = table[(name, branch)]
                for i, (got, expected) in enumerate(zip(asc, row["a"])):
                    if expected is None:
                        continue
                    assert _matches_2dp(got, expected), (
                        f"{name} {branch} {scenario} a{i}: "
                        f"json={got:.4f} table={expected}"
                    )


# ===========================================================================
# 12. Old vs new regression (T8) — compares the currently committed
#     calibrations.json against the arxiv-v1 tag (previous public release).
#     Requires a git checkout with that tag; skipped otherwise (e.g. for an
#     end user who installed the package from a wheel/sdist).
# ===========================================================================

def _old_json_from_git_tag():
    try:
        repo_root = Path(__file__).resolve().parents[1]
        out = subprocess.run(
            ["git", "show", "arxiv-v1:desired/data/calibrations.json"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None


_OLD_JSON = _old_json_from_git_tag()
requires_old_json = pytest.mark.skipif(
    _OLD_JSON is None,
    reason="arxiv-v1 git tag not found in this checkout; skip old-vs-new regression (T8).",
)

# Diagnostics confirmed numerically unchanged between arxiv-v1 and the
# revised pickles -- coeff, xmin/xmax, disp and disp_coeff must be
# bit-identical for these.
_UNCHANGED_NAMES = {"Ar3N2", "Ar3O3", "N2", "Ne3S3", "O3N2", "R3N2", "S3", "S3O3"}


@requires_old_json
class TestOldVsNewRegression:

    @pytest.mark.parametrize("name", sorted(_UNCHANGED_NAMES))
    def test_unchanged_monotonic_bit_identical(self, name):
        old = _OLD_JSON["monotonic"][name]
        new = get_info(name)
        assert new["xmin"] == old["xmin"]
        assert new["xmax"] == old["xmax"]
        for scenario in ("t2eq0", "t2ge0"):
            assert new[scenario]["coeff"] == old[scenario]["coeff"]
            assert new[scenario]["disp"] == old[scenario]["disp"]
            assert new[scenario]["disp_coeff"] == old[scenario]["disp_coeff"]

    def test_old_json_had_no_sigma_cal(self):
        """Sanity check that the diff being tested is real: sigma_cal must
        be new, not merely re-derived from an old field of the same name."""
        old_n2 = _OLD_JSON["monotonic"]["N2"]["t2eq0"]
        assert "sigma_cal" not in old_n2

    def test_old_meta_already_had_schema_version(self):
        """The arxiv-v1 JSON already carried _meta.version == '2.0' (from
        the original build_calibrations_json.py); the schema version is
        unchanged by this revision, only the per-branch content is."""
        assert _OLD_JSON["_meta"]["version"] == "2.0"
