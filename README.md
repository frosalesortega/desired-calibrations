# DESIRED strong-line metallicity calibrations

Python package for deriving gas-phase oxygen abundances (12 + log(O/H)) from optical strong emission-line ratios, based on the DEep Spectra of Ionised REgions Database (DESIRED), the largest compilation of H II regions and galaxies with direct electron-temperature determinations assembled to date.

> **Reference:** 

 - [The DESIRED strong-line calibrations: I. New empirical metallicity relations for the local and high-redshift universe](https://arxiv.org/abs/2604.16273)


## Features

- **27 calibrations**: 19 monotonic + 8 double-valued strong-line indices, covering oxygen-, nitrogen-, sulphur-, argon-, and neon-based line ratios.
- **Two abundance scenarios**: *t*² = 0 (no temperature fluctuations) and *t*² ≥ 0 (temperature inhomogeneities).
- **Monte Carlo error propagation**: fully accounts for observational uncertainties.
- **Index-dependent dispersion**: σ(*x*) profile provides local calibration uncertainty at each index value, rather than a single global value.
- **Single function API**: one `calibrate()` call for any index.
- **NumPy only**: no exotic dependencies.

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
OH_t2eq0, err, OH_t2ge0, err2 = calibrate('R3', 0.3, index_e=0.1, branch='upper')

# Or get both branches at once
(OH_up, err_up, OH_t2_up, err_t2_up,
 OH_lo, err_lo, OH_t2_lo, err_t2_lo) = calibrate('R3', 0.3, index_e=0.1)
```

## API Reference

### `calibrate(name, index, index_e=None, *, include_disp=True, disp_mode='max', branch=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Calibration name (e.g., `'N2'`, `'O3N2'`, `'R3'`) |
| `index` | float or array | Observed line-ratio index value(s) |
| `index_e` | float, array, or None | 1σ error(s). If `None`, returns calibration dispersion only |
| `include_disp` | bool | Add calibration dispersion in quadrature to MC errors (default: `True`) |
| `disp_mode` | str | Dispersion mode: `'global'`, `'local'`, or `'max'` (default: `'max'`; see below) |
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

1. Calibration parameters (polynomial coefficients, dispersion profile, validity ranges) are stored in `calibrations.json`.
2. The `calibrate()` function evaluates the polynomial and propagates errors via Monte Carlo simulation (1000 realisations by default).
3. Values outside the validity range return `NaN`.

## Error Propagation and Calibration Dispersion

When `index_e` is provided, errors are estimated using Monte Carlo propagation: *N* random realisations of the index are drawn from 𝒩(index, index_e²), the polynomial is evaluated at each realisation, and the standard deviation of the resulting abundances gives the propagated measurement error σ<sub>MC</sub>.

If `include_disp=True` (default), the calibration dispersion σ<sub>cal</sub> is added in quadrature:

> σ<sub>total</sub> = √(σ<sub>MC</sub>² + σ<sub>cal</sub>²)

The `disp_mode` parameter controls how σ<sub>cal</sub> is evaluated:

| Mode | σ<sub>cal</sub> | Description |
|------|---------|-------------|
| `'global'` | σ<sub>fit</sub> | A single constant value (RMS scatter of the calibration residuals) applied uniformly across the full index range. |
| `'local'`  | σ(*x*) | An index-dependent dispersion profile evaluated at each data point, computed from the quadratic parametrisation σ(*x*) = *q*₀ + *q*₁*x* + *q*₂*x*² (Eq. 3 of Rosales-Ortega et al. 2026a). This captures the fact that calibration uncertainty varies across the index range. |
| `'max'`    | max[σ<sub>fit</sub>, σ(*x*)] | The recommended default. Uses whichever is larger at each point, ensuring that the uncertainty is never underestimated in sparsely sampled regimes (see Sec. 4.4 of Rosales-Ortega et al. 2026a). |

The σ<sub>cal</sub> value represents the minimum 1σ uncertainty that should be assigned to any individual strong-line metallicity estimate, as it reflects the intrinsic scatter due to variations in ionisation parameter, electron density, and other physical conditions at fixed metallicity. This systematic uncertainty cannot be reduced by improving the signal-to-noise ratio of the observations.

```python
# Compare dispersion modes
OH, err_max, _, _   = calibrate('N2', -1.5, index_e=0.05, disp_mode='max')    # recommended
OH, err_glob, _, _  = calibrate('N2', -1.5, index_e=0.05, disp_mode='global')
OH, err_local, _, _ = calibrate('N2', -1.5, index_e=0.05, disp_mode='local')
```

If `index_e` is not provided, the returned errors correspond to the calibration dispersion only (evaluated according to `disp_mode`).

## Examples

The `examples/` directory contains scripts demonstrating how to use the package:

### `example_usage.py`

Complete usage examples covering all common scenarios:

```bash
python examples/example_usage.py
```

This script demonstrates scalar and array inputs, monotonic and double-valued calibrations, error propagation with the three dispersion modes, and how to inspect calibration parameters with `get_info()`.

### `plot_calibrations.py`

Diagnostic visualization of the calibration polynomials with MC error bands. Requires `matplotlib`.

```bash
# Plot all calibrations (overview mosaic + individual figures)
python examples/plot_calibrations.py

# Plot specific calibrations only
python examples/plot_calibrations.py N2 O3N2 R3

# Save to PDF instead of displaying on screen
python examples/plot_calibrations.py --save

# Combine both
python examples/plot_calibrations.py N2 R3 --save
```

Each figure shows the calibration polynomial (solid line), the MC-propagated 1σ error band, and the total error band (MC + calibration dispersion using `disp_mode='max'`) for the *t*² = 0 and *t*² ≥ 0 scenarios. When available, the index-dependent dispersion profile σ(*x*) is shown on the secondary axis. Double-valued calibrations display four panels (upper/lower branch × *t*² = 0 / *t*² ≥ 0).

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## File Structure

```
desired-calibrations/
├── desired/
│   ├── __init__.py            # Public API
│   ├── calibrate.py           # Core calibration engine
│   └── data/
│       └── calibrations.json  # All calibration parameters
├── examples/
│   ├── example_usage.py       # Usage examples
│   └── plot_calibrations.py   # Diagnostic plots
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
      title={The DESIRED strong-line calibrations: I. New empirical metallicity relations for the local and high-redshift universe}, 
      author={F. F. Rosales-Ortega and J. E. Méndez-Delgado and J. U. Guerrero-González and C. Esteban and J. García-Rojas and K. Z. Arellano-Córdova and A. Z. Lugo-Aranda and O. Espíndola-Camacho and J. C. López-Gutiérrez and L. E. Martínez-Rivero and C. Morisset and M. Orte-García and E. Reyes-Rodríguez and L. Toribio San Cipriano and K. Kreckel and O. Egorov and I. A. Zinchenko and S. F. Sánchez and J. M. Vílchez},
      year={2026},
      eprint={2604.16273},
      archivePrefix={arXiv},
      primaryClass={astro-ph.GA},
      url={https://arxiv.org/abs/2604.16273}, 
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
