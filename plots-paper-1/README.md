# Individual calibrator figures

Standalone, one-diagnostic-per-file versions of the multi-panel figures in Rosales-Ortega et al. (2026), "The DESIRED strong-line calibrations: I." Provided for convenience (e.g. for use in talks and presentations) — the paper itself remains the citable reference for all scientific content.

`<NAME>` follows the diagnostic labels used throughout the paper and this package (e.g. `R2`, `O3N2`, `R_hat`; see `list_calibrations()`).

## `calibration_curves/` — `calib_<NAME>.pdf` (27 files)

The DESIRED 12+log(O/H)–index relation for one diagnostic: data points colour/marker-coded by region type (H II regions, local and high-*z* star-forming galaxies), the *t*²=0 and *t*²>0 fits, the log(*U*) colour scale, object counts, and a secondary solar-metallicity axis.

## `accuracy_dispersion/` — `dispersion_<NAME>.pdf` and `sigma_<NAME>.pdf` (54 files)

Predicted vs. true (direct-method) 12+log(O/H) for *t*²=0 and *t*²>0 respectively, with a residual-histogram inset (split by branch for bivalued diagnostics) quantifying the calibration's scatter.

## `literature_comparisons/` — `comparison_<NAME>.pdf` (23 files)

The DESIRED calibration overlaid on the data density and on every available literature calibration for that diagnostic, each with its valid index range marked.

## Citation

If you use these figures, please cite the paper (see the main [README](../README.md#citation)).
