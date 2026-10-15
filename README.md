# DESIRED strong-line metallicity calibrations

Python package for deriving gas-phase oxygen abundances (12 + log(O/H)) from optical strong emission-line ratios, based on the DEep Spectra of Ionised REgions Database (DESIRED), the largest compilation of H II regions and galaxies with direct electron-temperature determinations assembled to date.

> **Reference:** 

 - [The DESIRED strong-line calibrations: I. New empirical metallicity relations for the local and high-redshift universe](https://arxiv.org/abs/2604.16273)


## Features

- **27 calibrations**: 19 monotonic + 8 double-valued strong-line indices, covering oxygen-, nitrogen-, sulphur-, argon-, and neon-based line ratios.
- **Two abundance scenarios**: *t*² = 0 (no temperature fluctuations) and *t*² ≥ 0 (temperature inhomogeneities).
- **Monte Carlo error propagation**: fully accounts for observational uncertainties.
- **σ<sub>cal</sub>-based uncertainty**: the recommended calibration dispersion (Sec. 4.1/4.4), with σ<sub>fit</sub> available via `sigma_source='fit'`.
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

### `calibrate(name, index, index_e=None, *, include_disp=True, disp_mode='max', sigma_source='cal', branch=None, mc_N=1000, mc_seed=2025)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Calibration name (e.g., `'N2'`, `'O3N2'`, `'R3'`) |
| `index` | float or array | Observed line-ratio index value(s) |
| `index_e` | float, array, or None | 1σ error(s). A scalar broadcasts against an array `index`. If `None`, returns calibration dispersion only |
| `include_disp` | bool | Add calibration dispersion in quadrature to MC errors (default: `True`) |
| `disp_mode` | str | Dispersion mode: `'global'`, `'local'`, or `'max'` (default: `'max'`; see below) |
| `sigma_source` | str | Which global dispersion `disp_mode='global'`/`'max'` uses: `'cal'` (σ<sub>cal</sub>, default) or `'fit'` (σ<sub>fit</sub>); see below |
| `branch` | str or None | For double-valued: `'upper'`, `'lower'`, or `None` (both) |
| `mc_N` | int | Number of Monte Carlo realisations (default: `1000`) |
| `mc_seed` | int | Random seed for MC reproducibility (default: `2025`) |

**Returns:** `(OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e)` — values where the input `index` falls outside the branch's validity range (`xmin`/`xmax`, printed as the Validity columns of Tables 3/4 of the reference) are `NaN`, and a `UserWarning` is raised whenever this happens for at least one input value.

For double-valued calibrations with `branch=None`, returns 8 values: `(upper_t2eq0, upper_t2eq0_e, upper_t2ge0, upper_t2ge0_e, lower_t2eq0, lower_t2eq0_e, lower_t2ge0, lower_t2ge0_e)`.

### `list_calibrations(kind=None)`

List available calibration names. Filter with `kind='monotonic'` or `kind='double_valued'`.

### `get_info(name)`

Return the full parameter dictionary for a given calibration: coefficients, dispersion (`disp`/σ<sub>fit</sub> and `sigma_cal`/σ<sub>cal</sub>), σ(*x*) profile coefficients (`disp_coeff`), validity range, and — for double-valued diagnostics — the informational `turnover`, `excluded_interval_OH`, and `notes` fields described in [Guidelines and caveats](#guidelines-and-caveats) below. None of the informational fields affect `calibrate()`'s output; they exist purely for inspection and documentation.

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

If `include_disp=True` (default), a global calibration dispersion σ<sub>global</sub> is added in quadrature:

> σ<sub>total</sub> = √(σ<sub>MC</sub>² + σ<sub>global</sub>²)

Two independent parameters control how σ<sub>global</sub> is evaluated: `disp_mode` chooses *where* it is evaluated (constant vs. index-dependent), and `sigma_source` chooses *which* quantity is used as the constant/floor term.

**`disp_mode`** (Sec. 4.4 of Rosales-Ortega et al. 2026):

| Mode | Description |
|------|-------------|
| `'global'` | A single constant value (selected by `sigma_source`) applied uniformly across the full index range. |
| `'local'`  | An index-dependent dispersion profile σ(*x*), evaluated at each data point as `numpy.polyval(disp_coeff, x)` — i.e. `disp_coeff[0]·x² + disp_coeff[1]·x + disp_coeff[2]` (descending power). **Note:** Eq. 3 of Rosales-Ortega et al. 2026 writes this profile as σ(*x*) = *e*₀ + *e*₁*x* + *e*₂*x*² (ascending power); the coefficients as printed in Tables 3/4 and stored here only give a non-negative, physically sensible σ(*x*) under the descending-power reading used by this package — this is a labelling issue in the manuscript's Eq. 3, not in the tabulated values. |
| `'max'`    | The recommended default. Uses max[σ<sub>global</sub>, σ(*x*)] at each point, ensuring that the uncertainty is never underestimated in sparsely sampled regimes. |

**`sigma_source`** (only relevant for `disp_mode='global'`/`'max'`; has no effect on `'local'`):

| Value | Quantity | Description |
|-------|----------|--------------|
| `'cal'` (default) | σ<sub>cal</sub> | The standard deviation of the metallicity residuals relative to the *T*<sub>e</sub>-based abundances (Sec. 4.1/4.4). This is the recommended, and generally larger, estimate of calibration scatter. |
| `'fit'` | σ<sub>fit</sub> | The RMS scatter of the polynomial fit residuals in line-ratio space. |

σ<sub>cal</sub> and σ<sub>fit</sub> are nearly identical for the 19 monotonic diagnostics (differences ≲0.004 dex), but differ substantially for the 8 double-valued diagnostics (up to ~0.07 dex, i.e. up to a third of the total uncertainty) — see `get_info(name)['t2eq0']['sigma_cal']` vs. `['disp']` (or, for a double-valued diagnostic, `get_info(name)['upper']['t2eq0']['sigma_cal']` vs. `['disp']`).

σ<sub>cal</sub> (and, in `'max'` mode, its comparison against σ(*x*)) represents the minimum 1σ uncertainty that should be assigned to any individual strong-line metallicity estimate, as it reflects the intrinsic scatter due to variations in ionisation parameter, electron density, and other physical conditions at fixed metallicity. This systematic uncertainty cannot be reduced by improving the signal-to-noise ratio of the observations (Sec. 4.4).

```python
# Compare dispersion modes and sigma sources
OH, err_max, _, _   = calibrate('N2', -1.5, index_e=0.05, disp_mode='max')    # recommended
OH, err_glob, _, _  = calibrate('N2', -1.5, index_e=0.05, disp_mode='global')
OH, err_local, _, _ = calibrate('N2', -1.5, index_e=0.05, disp_mode='local')
OH, err_fit,  _, _  = calibrate('N2', -1.5, index_e=0.05, disp_mode='global', sigma_source='fit')
```

If `index_e` is not provided, the returned errors correspond to the calibration dispersion only (evaluated according to `disp_mode`/`sigma_source`).

## Guidelines and caveats

The following points summarise recommendations from Sec. 4.4 ("Guidelines for the application of the DESIRED calibrations") of Rosales-Ortega et al. 2026. They are documentation only — `calibrate()` does not enforce them at runtime; interpreting the results correctly is left to the user.

**Proximity to the turnover of a double-valued calibration.** Near the turnover metallicity separating the upper and lower branches, the line ratio becomes increasingly insensitive to O/H — an intrinsic degeneracy of the diagnostic, not a fitting artefact — and the tabulated polynomial does not reproduce the vertical-tangent behaviour of the exact branch inversion there (Appendix C). Neither σ<sub>cal</sub> nor σ(*x*) fully captures the true growth of the uncertainty in this regime; metallicities inferred close to the turnover should be interpreted with particular caution. Per-branch/scenario turnover metallicities are available via `get_info(name)['turnover']`.

**Two distinct "gaps" for double-valued diagnostics — do not conflate them.** For an independent-branch-fit diagnostic (all bivalued diagnostics except R3 and R3S2), the upper and lower branches have two closely-spaced but distinct turnover metallicities, with the metallicity interval between them excluded from both fits (`get_info(name)['excluded_interval_OH']`; `get_info(name)['turnover']['shared']` is `False`). For R3 and R3S2, a single Direction-B fit reproduces both branches with one shared turnover, so this excluded interval is `0.0`. Separately, and for **all 8** double-valued diagnostics including R3/R3S2, there is a gap between the lower branch's upper O/H limit and the upper branch's lower O/H limit (`get_info(name)['lower']['t2eq0']['oh_max']` vs. `['upper']['t2eq0']['oh_min']`) — this reflects where the calibrating sample has data for each branch, not the turnover itself, and is typically 0.2–0.5 dex wide. Example (R23, *t*²=0): excluded interval (turnover-based) = 0.25 dex; branch-limit gap = 0.35 dex — similar in size here, but conceptually distinct and not always comparable.

**Ar3 upper branch.** Unlike every other double-valued diagnostic, the Ar3 upper branch is populated by too few calibrating regions to show the classical double-branch morphology; it is calibrated with a direct fit to binned medians rather than as the exact numerical inverse of a Direction-B relation, and its upper turnover (8.48 dex, *t*²=0) is adopted by convention as the lower turnover (8.22 dex) + 0.26 dex, rather than derived analytically (Appendix C). See `get_info('Ar3')['notes']['upper_branch']`.

**No explicit DIG or N/O correction.** These calibrations are purely empirical mappings between observed line ratios and *T*<sub>e</sub>-based O/H, with no explicit correction for diffuse ionised gas (DIG) contamination or for deviations from the mean N/O–O/H relation of the calibrating sample; both effects are folded into the empirical scatter (σ<sub>cal</sub>) rather than corrected for individually (Sec. 3.1). Nitrogen-based diagnostics (N2, O3N2, N2O2, N2S2, N2S2Hα, Ar3N2) should be avoided in favour of non-nitrogen alternatives where available, particularly above 12+log(O/H) ≈ 8.5 (Sec. 4.4).

## Applicability at high redshift

Sec. 4.3 of Rosales-Ortega et al. 2026 compares the *z*>2 subsample of the DESIRED calibration sample against the full sample, per diagnostic and branch, and classifies each into one of four categories:

1. **No significant offset:** R2, O32, R2Ne3, O3N2, R3N2, S3, S23, and the lower branches of R̂ and RNe.
2. **Positive offset** (consistent with elevated N/O at *z*>2): N2, N2O2, N2S2, N2S2Hα, and the low-metallicity branches of R3 and R23.
3. **Offset of opposite sign** (consistent with harder ionising conditions at *z*>2): Ne3O2, O3S2, Ne3S2, S2, and the high-metallicity branches of R3 and R3S2.
4. **Not assessable** with the current *z*>2 subsample (insufficient coverage within the formally valid range): the upper branches of R23, R̂, and RNe, and the lower branch of Ar3.

Diagnostics/branches not listed above were not part of this specific *z*>2 assessment. See Sec. 4.3 and Fig. 13 of the reference for the full statistical methodology and the corresponding offset magnitudes.

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

A subset of tests cross-checks the package against the private calibration pickles and the manuscript's LaTeX source tables; these are not part of the public package (see `.gitignore`) and are skipped by default. To run them, set:

```bash
DESIRED_TEST_BIN_DIR=/path/to/bin DESIRED_TEST_PAPER_DIR=/path/to/paper pytest tests/ -v
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

## Changelog

### v2.0.0

This release incorporates the calibrations as revised during peer review for Rosales-Ortega et al. (2026). Coefficients, validity ranges and dispersions were regenerated for all 27 diagnostics from the revised fits; eight diagnostics (Ar3N2, Ar3O3, N2, Ne3S3, O3N2, R3N2, S3, S3O3) remain numerically identical to the initial release.

The most consequential change for existing users concerns the reported uncertainties: `calibrate()`'s default now uses σ<sub>cal</sub> — the scatter of the calibration residuals relative to the *T*<sub>e</sub>-based abundances — rather than σ<sub>fit</sub>, the polynomial fit residual (either can still be selected via the new `sigma_source` parameter). The difference is negligible for the monotonic diagnostics but reaches up to ~0.07 dex for the double-valued ones, so uncertainties from this version are not directly comparable to the previous release.

`get_info()` also reports a few additional, purely informational fields — per-branch `oh_min`/`oh_max`, and for double-valued diagnostics the branch `turnover` metallicity and `excluded_interval_OH` — with no effect on `calibrate()`'s validity (NaN) logic, which remains based solely on the line-ratio index. Out-of-range index values now consistently raise a `UserWarning`. Two minor issues present since the initial release were also corrected: `list_calibrations(kind=...)` validates its argument instead of silently returning an empty list, and `index_e` broadcasts against an array `index` following standard NumPy rules. The `calibrations.json` schema version was bumped to `2.1` to reflect the additive fields above.

## Citation

If you use this package, please cite:

```bibtex
@article{RosalesOrtega2026,
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
