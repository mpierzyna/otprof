# WRF simulation — Canary Islands

Workspace to run twelve 5-day WRF simulations (one per month of 2017) over the Canary Islands using
[`wrf_massive`](https://github.com/mpierzyna/wrf-massive). Forcing is downloaded from the Copernicus
Climate Data Store (CDS) or the TU Delft CERRA mirror; stages run locally (chaos) or split across DelftBlue (WPS) and
Snellius (WRF + post-proc).

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

Twelve 5-day net runs in 2017, one per month (**2017-MM-01 → 2017-MM-06**), each with a **12-hour spin-up**
prepended. Defined in `simulations.py` as `sim_canaries`; each becomes a `sim_2017-MM-01/` directory with the
pipeline stages below.

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
sim_2017-MM-01/
├── 1_forcing/     # CERRA + ERA5 grib from CDS (deleted after WPS)
├── 2_wps/         # WPS output: met_em*.nc files (deleted after post-proc)
├── 3_wrf/         # WRF run directory and wrfout files
└── 4_postproc/    # cn2 NetCDF files (final output)
```

Intermediate data are garbage-collected automatically: grib files are removed once WPS completes; `met_em*.nc`
files are removed once Cn² output exists.

## Machine configurations

Select the active machine by symlinking `env.yaml` (read by `cli.py`) to one of the environment files:

```bash
ln -sf env_<machine>.yaml env.yaml
```

| Machine | File | Pipeline | Purpose |
|---|---|---|---|
| `chaos` | `env_chaos.yaml` | `p_default` | Full pipeline, forcing from TU Delft CERRA mirror |
| `chaos` | `env_chaos_cds.yaml` | `p_cds` | Full pipeline, forcing from CDS |
| `delftblue` | `env_delftblue.yaml` | `p_preproc` | Pull CERRA from mirror + WPS (dmpar, 4 tasks) |
| `snellius` | `env_snellius.yaml` | `p_snellius` | WRF (32 MPI tasks, 12 h) + Cn² post-proc on scratch |

### Typical two-machine workflow

1. **DelftBlue**: run `p_preproc` to pull CERRA and produce `met_em*.nc` files.
2. **Snellius**: run `p_snellius` to execute WRF and post-processing on `/scratch-shared`, then copy results back.

## Running

```bash
# Create the sim dirs + simulation.yaml
uv run python cli.py init-sims simulations.py sim_canaries

# Run a single simulation (or a subset of stages)
uv run python cli.py run --stages=<stage> sim_2017-01-01

# Submit a SLURM array job (reads sim dirs from .array_sim_dirs)
sbatch --array=1-N slurm_<machine>.job.sh <stage>
```

`wrf_massive` tracks completion via `.done` marker files; finished stages are skipped on re-run.
