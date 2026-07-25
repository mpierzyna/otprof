# WRF simulation — Canary Islands

Workspace to run a single 5-day WRF simulation over the Canary Islands using
[`wrf_massive`](https://github.com/mpierzyna/wrf-massive). Forcing is downloaded directly from the Copernicus
Climate Data Store (CDS); all stages run locally on one machine (chaos).

## Domain & grid

| Parameter | Value |
|---|---|
| Projection | Lambert conformal (tangent, truelat 28.4°) |
| Center | 28.4°N, 15.7°W |
| Grid size | 280 × 150 (W–E × S–N) |
| Horizontal resolution | 2 km (~558 × 298 km, covers all seven islands) |
| Vertical levels | 101 eta levels |
| Model top | 1000 Pa |
| Time step | 10 s |

## Simulation

One 5-day net run, **2020-07-01 → 2020-07-06** (summer trade-wind regime), with a **12-hour spin-up** prepended.
Defined in `simulations.py` as `sim_canaries`; it becomes a `sim_2020-07-01/` directory with the pipeline stages
below.

## Physics

**PBL / surface layer:** MYNN 2.5 (`bl_pbl_physics = 5`, `sf_sfclay_physics = 5`) without EDMF
(`bl_mynn_edmf = 0`), mixing-length option 1. TKE advection and budget diagnostics enabled.

| Scheme | Option |
|---|---|
| Microphysics | Lin et al. (option 4) |
| LW / SW radiation | RRTMG (option 4) |
| Land surface | Noah (option 2) |
| Cumulus | Kain–Fritsch (option 1) |
| Urban | disabled |

## Forcing data (CDS)

Lateral boundary conditions come from **CERRA reanalysis** (3-hourly), with **ERA5** filling the fields CERRA
does not carry. WPS ungribs both and metgrid merges them (`fg_name = 'ERA5','CERRA'`, CERRA takes precedence).

| Source | CDS dataset | Provides |
|---|---|---|
| CERRA | `reanalysis-cerra-pressure-levels` | T, U, V, RH, geopotential on 29 pressure levels |
| CERRA | `reanalysis-cerra-single-levels` | 2m T/RH, surface & MSL pressure, land-sea mask, orography, skin temp, snow depth |
| ERA5 | `reanalysis-era5-single-levels` | 4-layer soil temperature/moisture + 10m wind (u/v) |

ERA5 supplies 10m wind because CERRA archives it only as speed/direction, which ungrib cannot convert to the
u/v components WPS needs. CERRA's projected grid is downloaded at full extent (no CDS geographic subsetting) and
cropped by WPS; ERA5 is cropped to the simulation `area`.

**Credentials:** the download uses the `cdsapi` client, which reads `~/.cdsapirc` (or `CDSAPI_URL`/`CDSAPI_KEY`).

## Output

**Main output** (`wrfout_d01_*`): hourly, all frames in one file. **Auxiliary** (`wrfout_aux_d01_*`): hourly,
6-frame files. Extra fields beyond WRF defaults are in `myoutfields.txt`.

Post-processing (`4_postproc/`) computes Cn² and extracts variables into compressed NetCDF.

## Pipeline stages

```
sim_2020-07-01/
├── 1_forcing/     # CERRA + ERA5 grib from CDS (deleted after WPS)
├── 2_wps/         # WPS output: met_em*.nc files (deleted after post-proc)
├── 3_wrf/         # WRF run directory and wrfout files
└── 4_postproc/    # cn2 NetCDF files (final output)
```

Intermediate data are garbage-collected automatically: grib files are removed once WPS completes; `met_em*.nc`
files are removed once Cn² output exists.

## Running

The machine profile is `env_chaos.yaml`, symlinked to `env.yaml` (read by `cli.py`). Everything runs locally:

```bash
# Create the sim dir + simulation.yaml
uv run python cli.py init-sims simulations.py sim_canaries

# Run the whole pipeline (or a subset of stages)
uv run python cli.py run sim_2020-07-01
uv run python cli.py run --stages cerra,era5 sim_2020-07-01
```

`wrf_massive` tracks completion via `.done` marker files; finished stages are skipped on re-run.

Machine paths (compiled WPS/WRF, geog data) are set in `env_chaos.yaml`.
