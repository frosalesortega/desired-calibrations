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
print("Example: O3N2 calibration (array input)")
print("=" * 60)

O3N2_values = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
O3N2_errors = np.full_like(O3N2_values, 0.1)

OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = calibrate(
    "O3N2", O3N2_values, index_e=O3N2_errors
)

print(f"  {'O3N2':>8s}  {'OH(t2=0)':>10s}  {'err':>6s}  {'OH(t2>0)':>10s}  {'err':>6s}")
for i in range(len(O3N2_values)):
    print(f"  {O3N2_values[i]:8.2f}  {OH_t2eq0[i]:10.2f}  {OH_t2eq0_e[i]:6.2f}"
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
# 5. Double-valued calibration — specific branch
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: R3 (double-valued), upper branch")
print("=" * 60)

OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = calibrate(
    "R3", 0.6, index_e=0.1, branch="upper"
)
print(f"  R3 = 0.6 ± 0.1 (upper branch)")
print(f"  12+log(O/H) [t2=0]:   {OH_t2eq0:.2f} ± {OH_t2eq0_e:.2f}")
print(f"  12+log(O/H) [t2>=0]:  {OH_t2ge0:.2f} ± {OH_t2ge0_e:.2f}")
print()

# -----------------------------------------------------------------------
# 6. Double-valued calibration — both branches
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: R3 (double-valued), both branches")
print("=" * 60)

result = calibrate("R3", 0.6, index_e=0.1)
(OH_up_t2eq0, OH_up_t2eq0_e, OH_up_t2ge0, OH_up_t2ge0_e,
 OH_lo_t2eq0, OH_lo_t2eq0_e, OH_lo_t2ge0, OH_lo_t2ge0_e) = result

print(f"  R3 = 0.6 ± 0.1")
print(f"  Upper branch:")
print(f"    12+log(O/H) [t2=0]:   {OH_up_t2eq0:.2f} ± {OH_up_t2eq0_e:.2f}")
print(f"    12+log(O/H) [t2>=0]:  {OH_up_t2ge0:.2f} ± {OH_up_t2ge0_e:.2f}")
print(f"  Lower branch:")
print(f"    12+log(O/H) [t2=0]:   {OH_lo_t2eq0:.2f} ± {OH_lo_t2eq0_e:.2f}")
print(f"    12+log(O/H) [t2>=0]:  {OH_lo_t2ge0:.2f} ± {OH_lo_t2ge0_e:.2f}")
print()

# -----------------------------------------------------------------------
# 7. Get calibration info
# -----------------------------------------------------------------------
print("=" * 60)
print("Example: Inspect calibration parameters")
print("=" * 60)

info = get_info("N2")
print(f"  Name:       N2")
print(f"  Type:       {info['type']}")
print(f"  Definition: {info['definition']}")
print(f"  Range:      [{info['xmin']}, {info['xmax']}]")
print(f"  t2eq0 coeff: {info['t2eq0']['coeff']}")
print(f"  t2eq0 disp:  {info['t2eq0']['disp']}")
