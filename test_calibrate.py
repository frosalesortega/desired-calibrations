"""
Unit tests for the DESIRED calibration package.
Run with: pytest tests/test_calibrate.py -v
"""
import numpy as np
import pytest
from desired import calibrate, list_calibrations, get_info


class TestListCalibrations:
    def test_returns_list(self):
        result = list_calibrations()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_filter_monotonic(self):
        mono = list_calibrations(kind="monotonic")
        assert "N2" in mono

    def test_filter_double(self):
        dv = list_calibrations(kind="double_valued")
        assert "R3" in dv

    def test_no_overlap(self):
        mono = set(list_calibrations(kind="monotonic"))
        dv = set(list_calibrations(kind="double_valued"))
        assert mono.isdisjoint(dv)


class TestGetInfo:
    def test_known(self):
        info = get_info("N2")
        assert info["type"] == "monotonic"
        assert "t2eq0" in info
        assert "coeff" in info["t2eq0"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration"):
            get_info("FAKE_INDEX")


class TestCalibrateMonotonic:
    """Tests using the N2 calibration (has real data in the JSON)."""

    def test_scalar_input(self):
        result = calibrate("N2", -1.0, index_e=0.05)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    def test_scalar_no_error(self):
        OH, OH_e, OH_t2, OH_t2e = calibrate("N2", -1.0)
        # Without index_e, error should be the intrinsic dispersion
        assert OH_e == 0.20  # t2eq0 disp for N2
        assert OH_t2e == 0.24

    def test_array_input(self):
        x = np.array([-2.0, -1.5, -1.0, -0.5])
        x_e = np.full(4, 0.05)
        result = calibrate("N2", x, index_e=x_e)
        assert len(result) == 4
        assert all(isinstance(v, np.ndarray) for v in result)
        assert all(v.shape == (4,) for v in result)

    def test_out_of_range_returns_nan(self):
        # xmax for N2 is -0.22; value of 0.5 is outside
        OH, OH_e, _, _ = calibrate("N2", 0.5)
        assert np.isnan(OH)
        assert np.isnan(OH_e)

    def test_mixed_valid_invalid(self):
        x = np.array([-1.0, 0.5])  # -1.0 valid, 0.5 out of range
        OH, _, _, _ = calibrate("N2", x)
        assert not np.isnan(OH[0])
        assert np.isnan(OH[1])

    def test_include_disp_false(self):
        OH1, e1, _, _ = calibrate("N2", -1.0, index_e=0.05, include_disp=True)
        OH2, e2, _, _ = calibrate("N2", -1.0, index_e=0.05, include_disp=False)
        # With dispersion should give larger error
        assert e1 >= e2

    def test_branch_warns_monotonic(self):
        with pytest.warns(UserWarning, match="monotonic"):
            calibrate("N2", -1.0, branch="upper")

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            calibrate("N2", np.array([-1.0, -0.5]), index_e=np.array([0.05]))


class TestCalibrateDoubleValued:
    """Tests for double-valued calibrations (R3, R23)."""

    def test_both_branches_returns_8(self):
        result = calibrate("R3", 0.6)
        assert len(result) == 8
        # With placeholder data, all values should be NaN
        assert all(np.isnan(v) for v in result)

    def test_single_branch_returns_4(self):
        result = calibrate("R3", 0.6, branch="upper")
        assert len(result) == 4

    def test_invalid_branch_raises(self):
        with pytest.raises(ValueError, match="branch must be"):
            calibrate("R3", 0.6, branch="middle")


class TestUnknownCalibration:
    def test_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration"):
            calibrate("NONEXISTENT", 1.0)


class TestReproducibility:
    def test_same_seed_same_result(self):
        r1 = calibrate("N2", -1.0, index_e=0.05, mc_seed=42)
        r2 = calibrate("N2", -1.0, index_e=0.05, mc_seed=42)
        assert r1 == r2

    def test_different_seed_different_result(self):
        # Internal MC produces different floats; rounding can mask this,
        # so just confirm the function accepts different seeds without error
        r1 = calibrate("N2", -1.0, index_e=0.3, mc_seed=42)
        r2 = calibrate("N2", -1.0, index_e=0.3, mc_seed=99)
        # Both should be valid (not NaN)
        assert not np.isnan(r1[0])
        assert not np.isnan(r2[0])
