# DESIRED — Strong-Line Metallicity Calibrations

**DEep Spectra of Ionised REgions Database**

Python package for deriving gas-phase oxygen abundances (12 + log(O/H)) from optical strong emission-line ratios, based on the most extensive compilation of *T*<sub>e</sub>-based literature data to date.

> **Reference:** Rosales-Ortega et al. (2026a)

## Features

- **27 calibrations**: 19 monotonic + 8 double-valued strong-line indices
- **Two abundance scenarios**: *t*² = 0 (no temperature fluctuations) and *t*² ≥ 0
- **Monte Carlo error propagation**: fully accounts for observational uncertainties
- **Intrinsic dispersion**: optionally included in error budget
- **Single function API**: one `calibrate()` call for any index
- **NumPy only**: no exotic dependencies

## Installation

```bash
# From GitHub
pip install git+https://github.com/frosalesortega/desired-calibrations.git

# Or clone and install locally
git clone https://github.com/frosalesortega/desired-calibrations.git
cd desired-calibrations
pip install .
```

## Quick Start

```python
import numpy as np
from desired import calibrate, list_calibrations

# See available calibrations
list_calibrations()
# ['Ar3', 'Ar3N2', 'Ar3O3', 'N2', 'N2O2', 'N2S2', 'N2S2Ha', 'Ne3', ...]

# --- Monotonic calibration ---
OH_t2eq0, err_t2eq0, OH_t2ge0, err_t2ge0 = calibrate('N2', -1.2, index_e=0.05)
print(f"12+log(O/H) = {OH_t2eq0:.2f} ± {err_t2eq0:.2f}  [t2=0]")
print(f"12+log(O/H) = {OH_t2ge0:.2f} ± {err_t2ge0:.2f}  [t2≥0]")

# --- Array input ---
N2_values = np.array([-2.0, -1.5, -1.0, -0.5])
N2_errors = np.array([ 0.1,  0.08, 0.05, 0.03])
OH_t2eq0, err_t2eq0, OH_t2ge0, err_t2ge0 = calibrate('N2', N2_values, index_e=N2_errors)

# --- Double-valued calibration (e.g., R3) ---
# Specify branch explicitly
OH_t2eq0, err, OH_t2ge0, err2 = calibrate('R3', 0.6, index_e=0.1, branch='upper')

# Or get both branches at once
(OH_up, err_up, OH_t2_up, err_t2_up,
 OH_lo, err_lo, OH_t2_lo, err_t2_lo) = calibrate('R3', 0.6, index_e=0.1)
```

## API Reference

### `calibrate(name, index, index_e=None, *, include_disp=True, branch=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Calibration name (e.g., `'N2'`, `'O3N2'`, `'R3'`) |
| `index` | float or array | Observed line-ratio index value(s) |
| `index_e` | float, array, or None | 1σ error(s). If `None`, returns intrinsic dispersion only |
| `include_disp` | bool | Add intrinsic dispersion in quadrature to MC errors (default: `True`) |
| `branch` | str or None | For double-valued: `'upper'`, `'lower'`, or `None` (both) |

**Returns:** `(OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e)` — values outside the validity range are `NaN`.

For double-valued calibrations with `branch=None`, returns 8 values: `(upper_t2eq0, upper_t2eq0_e, upper_t2ge0, upper_t2ge0_e, lower_t2eq0, lower_t2eq0_e, lower_t2ge0, lower_t2ge0_e)`.

### `list_calibrations(kind=None)`

List available calibration names. Filter with `kind='monotonic'` or `kind='double_valued'`.

### `get_info(name)`

Return the full parameter dictionary (coefficients, dispersion, validity range) for a given calibration.

## Available Calibrations

### Monotonic (19)

| Index | Definition |
|-------|-----------|
| R2 | log([O II]λ3727 / Hβ) |
| O32 | log([O III]λ5007 / [O II]λ3727) |
| Ne3O2 | log([Ne III]λ3869 / [O II]λ3727) |
| R2Ne3 | log(([O II]λ3727 + [Ne III]λ3869) / Hβ) |
| N2 | log([N II]λ6584 / Hα) |
| O3N2 | log(([O III]λ5007/Hβ) / ([N II]λ6584/Hα)) |
| R3N2 | log([O III]λ5007 / [N II]λ6584) |
| N2O2 | log([N II]λ6584 / [O II]λ3727) |
| S2 | log([S II]λ6717,6731 / Hα) |
| N2S2 | log([N II]λ6584 / [S II]λ6717,6731) |
| N2S2Ha | log([N II]λ6584/[S II]λ6717,6731) + 0.264×log([N II]λ6584/Hα) |
| S3 | log([S III]λ9069,9532 / Hα) |
| S3O3 | log([S III]λ9069 / [O III]λ5007) |
| S23 | log(([S II]λ6717,6731 + [S III]λ9069,9532) / Hα) |
| O3S2 | log(([O III]λ5007/Hβ) / ([S II]λ6717,6731/Hα)) |
| Ar3O3 | log([Ar III]λ7136 / [O III]λ5007) |
| Ar3N2 | log([Ar III]λ7136 / [N II]λ6584) |
| Ne3S2 | log([Ne III]λ3869 / [S II]λ6717,6731) |
| Ne3S3 | log([Ne III]λ3869 / [S III]λ9069) |

### Double-Valued (8)

| Index | Definition |
|-------|-----------|
| R3 | log([O III]λ5007 / Hβ) |
| R23 | log(([O II]λ3727 + [O III]λ4959,5007) / Hβ) |
| R̂ | 0.47×R2 + 0.88×R3 |
| RNe | 0.47×log([O II]λ3727/Hγ) + 0.88×log([Ne III]λ3869/Hγ) |
| Ne3 | log([Ne III]λ3869 / Hβ) |
| R3S2 | log([O III]λ5007/Hβ + [S II]λ6717,6731/Hα) |
| Ar3 | log([Ar III]λ7136 / Hα) |
| O3HeI | log([O III]λ5007 / He Iλ5876) |

## How It Works

1. Calibration parameters (polynomial coefficients, intrinsic dispersion, validity ranges) are stored in `calibrations.json`.
2. The `calibrate()` function evaluates the polynomial and propagates errors via Monte Carlo simulation (1000 realisations by default).
3. Values outside the validity range return `NaN`.

## Error Propagation

When `index_e` is provided, errors are estimated using Monte Carlo propagation:

- *N* random realisations of the index are drawn from 𝒩(index, index_e²)
- The polynomial is evaluated at each realisation
- The standard deviation of the resulting abundances gives the propagated error
- If `include_disp=True`, the intrinsic calibration dispersion σ<sub>int</sub> is added in quadrature:
  
  σ<sub>total</sub> = √(σ<sub>MC</sub>² + σ<sub>int</sub>²)

## File Structure

```
desired/
├── desired/
│   ├── __init__.py            # Public API
│   ├── calibrate.py           # Core calibration engine
│   └── data/
│       └── calibrations.json  # All calibration parameters
├── examples/
│   └── example_usage.py       # Usage examples
├── tests/
│   └── test_calibrate.py      # Unit tests
├── setup.py
├── LICENSE
└── README.md
```

## Citation

If you use this package, please cite:

```bibtex
@article{RosalesOrtega2026a,
    author  = {Rosales-Ortega, F. F. and others},
    title   = {DESIRED strong-line metallicity calibrations},
    journal = {},
    year    = {2026},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
