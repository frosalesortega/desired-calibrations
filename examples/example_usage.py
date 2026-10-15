"""
Example: Using the DESIRED strong-line metallicity calibrations
===============================================================

This script demonstrates how to use the ``desired`` package to derive
gas-phase oxygen abundances (12 + log(O/H)) from strong emission-line
ratios, for both the t^2 = 0 and t^2 >= 0 scenarios.

Reference: Rosales-Ortega et al. (2026a)
"""

import numpy as np
from desired import calibrate, list_calibrations, get_info

# -----------------------------------------------------------------------
# 1. List all available calibrations
# -----------------------------------------------------------------------
print("Available calibrations:")
print("  Monotonic:     ", list_calibrations(kind="monotonic"))
print("  Double-valued: ", list_calibrations(kind="double_valued"))
print()

# -----------------------------------------------------------------------
# 2. Monotonic calibration — single value
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: N2 calibration (scalar input)")
print("=" * 60)

N2_value = -1.2
N2_error = 0.05

OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = calibrate(
    "N2", N2_value, index_e=N2_error
)

print(f"  N2 = {N2_value} ± {N2_error}")
print(f"  12+log(O/H) [t2=0]:   {OH_t2eq0:.2f} ± {OH_t2eq0_e:.2f}")
print(f"  12+log(O/H) [t2>=0]:  {OH_t2ge0:.2f} ± {OH_t2ge0_e:.2f}")
print()

# -----------------------------------------------------------------------
# 3. Monotonic calibration — array of values
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: N2 calibration (array input)")
print("=" * 60)

N2_values = np.array([-2.5, -2.0, -1.5, -1.0, -0.5])
N2_errors = np.full_like(N2_values, 0.1)

OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = calibrate(
    "N2", N2_values, index_e=N2_errors
)

print(f"  {'N2':>8s}  {'OH(t2=0)':>10s}  {'err':>6s}  {'OH(t2>0)':>10s}  {'err':>6s}")
for i in range(len(N2_values)):
    print(f"  {N2_values[i]:8.2f}  {OH_t2eq0[i]:10.2f}  {OH_t2eq0_e[i]:6.2f}"
          f"  {OH_t2ge0[i]:10.2f}  {OH_t2ge0_e[i]:6.2f}")
print()

# -----------------------------------------------------------------------
# 4. Monotonic — without errors (returns dispersion only)
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: N2 without index errors (dispersion only)")
print("=" * 60)

OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = calibrate("N2", -1.0)
print(f"  N2 = -1.0")
print(f"  12+log(O/H) [t2=0]:   {OH_t2eq0:.2f} ± {OH_t2eq0_e:.2f} (disp)")
print(f"  12+log(O/H) [t2>=0]:  {OH_t2ge0:.2f} ± {OH_t2ge0_e:.2f} (disp)")
print()

# -----------------------------------------------------------------------
# 5. Dispersion modes: global vs local vs max
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: Dispersion modes (disp_mode)")
print("=" * 60)
print("  N2 = -1.0 ± 0.05")
print()

for mode in ("global", "local", "max"):
    OH, e, OH2, e2 = calibrate("N2", -1.0, index_e=0.05, disp_mode=mode)
    print(f"  disp_mode='{mode:6s}':  OH={OH:.2f} ± {e:.2f}  [t2=0]"
          f"    OH={OH2:.2f} ± {e2:.2f}  [t2>=0]")

print()
print("  'global' = a constant value across the entire range (sigma_cal by default)")
print("  'local'  = sigma(x), evaluated via np.polyval(disp_coeff, x)")
print("  'max'    = max[global value, sigma(x)] (recommended, default)")
print()

# Show how dispersion varies across the range
print("  Dispersion profile across N2 range:")
N2_range = np.array([-2.5, -2.0, -1.5, -1.0, -0.5])
for mode in ("global", "local", "max"):
    _, errs, _, _ = calibrate("N2", N2_range, disp_mode=mode)
    err_str = "  ".join(f"{e:.2f}" for e in errs)
    print(f"    {mode:6s}: [{err_str}]")
print()

# -----------------------------------------------------------------------
# 5b. sigma_source: sigma_cal (default, recommended) vs sigma_fit
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: sigma_source ('cal' vs 'fit')")
print("=" * 60)
print("  For most monotonic calibrations the two are nearly identical;")
print("  for double-valued calibrations they can differ by several")
print("  hundredths of a dex -- R_hat (lower, t2>=0) shown here:")
print()

for src in ("cal", "fit"):
    OH, e, OH2, e2 = calibrate("R_hat", 0.0, branch="lower",
                                disp_mode="global", sigma_source=src)
    print(f"  sigma_source='{src}':  OH={OH:.2f} ± {e:.2f}  [t2=0]"
          f"    OH={OH2:.2f} ± {e2:.2f}  [t2>=0]")
print()

# -----------------------------------------------------------------------
# 6. Double-valued calibration — specific branch
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: R3 (double-valued), upper branch")
print("=" * 60)

OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = calibrate(
    "R3", 0.3, index_e=0.1, branch="upper"
)
print(f"  R3 = 0.3 ± 0.1 (upper branch)")
print(f"  12+log(O/H) [t2=0]:   {OH_t2eq0:.2f} ± {OH_t2eq0_e:.2f}")
print(f"  12+log(O/H) [t2>=0]:  {OH_t2ge0:.2f} ± {OH_t2ge0_e:.2f}")
print()

# -----------------------------------------------------------------------
# 7. Double-valued calibration — both branches
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: R3 (double-valued), both branches")
print("=" * 60)

result = calibrate("R3", 0.3, index_e=0.1)
(OH_up_t2eq0, OH_up_t2eq0_e, OH_up_t2ge0, OH_up_t2ge0_e,
 OH_lo_t2eq0, OH_lo_t2eq0_e, OH_lo_t2ge0, OH_lo_t2ge0_e) = result

print(f"  R3 = 0.3 ± 0.1")
print(f"  Upper branch:")
print(f"    12+log(O/H) [t2=0]:   {OH_up_t2eq0:.2f} ± {OH_up_t2eq0_e:.2f}")
print(f"    12+log(O/H) [t2>=0]:  {OH_up_t2ge0:.2f} ± {OH_up_t2ge0_e:.2f}")
print(f"  Lower branch:")
print(f"    12+log(O/H) [t2=0]:   {OH_lo_t2eq0:.2f} ± {OH_lo_t2eq0_e:.2f}")
print(f"    12+log(O/H) [t2>=0]:  {OH_lo_t2ge0:.2f} ± {OH_lo_t2ge0_e:.2f}")
print()

# -----------------------------------------------------------------------
# 8. Get calibration info
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: Inspect calibration parameters")
print("=" * 60)

info = get_info("N2")
print(f"  Name:       N2")
print(f"  Type:       {info['type']}")
print(f"  Definition: {info['definition']}")
print(f"  Range:      [{info['xmin']}, {info['xmax']}]")
print(f"  t2eq0 coeff:      {info['t2eq0']['coeff']}")
print(f"  t2eq0 disp:       {info['t2eq0']['disp']}       (sigma_fit)")
print(f"  t2eq0 sigma_cal:  {info['t2eq0']['sigma_cal']}       (recommended)")
print(f"  t2eq0 disp_coeff: {info['t2eq0'].get('disp_coeff', 'N/A')}")
print(f"  t2eq0 oh range:   [{info['t2eq0']['oh_min']}, {info['t2eq0']['oh_max']}]  (informational)")
print()

# -----------------------------------------------------------------------
# 9. Double-valued calibration — turnover and excluded interval (informational)
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: Turnover metallicity for a bivalued diagnostic")
print("=" * 60)

info = get_info("R23")
print(f"  R23 turnover (t2=0): upper={info['turnover']['t2eq0']['upper_OH']}"
      f"  lower={info['turnover']['t2eq0']['lower_OH']}"
      f"  shared={info['turnover']['shared']}")
print(f"  Excluded interval (t2=0): {info['excluded_interval_OH']['t2eq0']:.2f} dex")
print("  (this is informational only; calibrate() does not use it -- NaN")
print("   is decided purely from each branch's index validity range)")
print()

info = get_info("Ar3")
print(f"  Ar3 upper-branch note: {info['notes']['upper_branch']}")
