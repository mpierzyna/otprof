# Paper: reproducing the comparisons

Configs and scripts behind the figures and tables of the paper (`<citation / DOI>`). They only use the otprof
package API and are not part of it: the package itself holds what live forecasting and new domains need (the
production model `examples/configs/wrf_pl_h.yaml`).

Data: the nl and canaries datasets in the layout of `examples/new_domain` (`4_features_interp/X_wrf_pl.nc`,
`3_features/X_era5_pl.nc`, `3_features/y_wrf.nc`), built with that example or downloaded from Zenodo (`<DOI>`). Ctl.
also needs `3_features/X_wrf_native.nc`: `make_inputs(wrf.standardize(ds, grid="model"), grid="model")` of the aligned
WRF output, not part of the example. Pass the domain directory with `--data-root`.

| Case in the paper | Config / script |
|---|---|
| WRF-PL, ERA5-PL (decoupled, the framework) | `configs/wrf_pl_h.yaml` |
| Ctl. (native WRF levels) | `configs/wrf_native_h.yaml` |
| ERA5-PL direct (appendix) | `configs/era5_pl_h_direct.yaml` |
| HV+W71 baseline | `hv_w71.py` |
| NL-oQM, canaries zero-shot and CI-oQM | follow with prediction and the output quantile mapping (migration step 8) |

Train: `uv run python examples/train.py <config> --data-root <dir>` (migration step 7).
