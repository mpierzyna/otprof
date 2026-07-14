# WRF simulation campaign — Netherlands domain

Workspace to generate ~1 year of WRF data over the Netherlands using [`wrf_massive`](https://github.com/mpierzyna/wrf-massive).

## Domain & grid

| Parameter | Value |
|---|---|
| Projection | Lambert conformal |
| Center | 52.109°N, 5.499°E (Netherlands) |
| Grid size | 144 × 192 (W–E × S–N) |
| Horizontal resolution | 2 km |
| Vertical levels | 101 eta levels |
| Model top | 1000 Pa |
| Time step | 10 s |

## Simulation design

365 days are covered in **73 non-overlapping 5-day chunks** (indices 0, 5, 10, … day-of-year), each with a **12-hour spin-up** period. The start years cycle through 2017–2020 so that the set samples all four seasons across multiple years.

Each chunk becomes a `sim_YYYY-MM-DD/` directory containing four pipeline stages.

## Physics

**PBL / surface layer:** MYNN 2.5 (`bl_pbl_physics = 5`, `sf_sfclay_physics = 5`) without EDMF (`bl_mynn_edmf = 0`) and with mixing-length option 1 (`bl_mynn_mixlength = 1`). TKE advection and budget diagnostics are enabled.

| Scheme | Option |
|---|---|
| Microphysics | Lin et al. (option 4) |
| LW / SW radiation | RRTMG (option 4) |
| Land surface | Noah (option 2) |
| Cumulus | Kain–Fritsch (option 1, outermost domain only) |
| Urban | disabled |

## Forcing data

Lateral boundary conditions are provided by **CERRA reanalysis** at 3-hourly intervals. WPS ungribs both ERA5 and CERRA (`fg_name = 'ERA5','CERRA'`).

The list of required CERRA files is tracked in `CERRA_files.txt.gz`. Files are pulled from a TU Delft remote (`tudelft:staff-umbrella/HBaki/CERRA`) via rclone.

## Output

**Main output** (`wrfout_d01_*`): hourly, all frames in a single file per chunk.

**Auxiliary output** (`wrfout_aux_d01_*`): hourly, 6-frame files.

Additional output fields beyond the WRF defaults are defined in `myoutfields.txt`:

```
z, HGT, U, V, W, T, P, PB, PBLH, PH, PHB, LANDMASK,
QKE, EL_PBL, TSQ, QVAPOR, QRAIN, QSNOW,
XLAT, XLONG, T2, TH2, U10, V10, LH, HFX, ZNT, UST, Z0
```

Post-processing (`4_postproc/`) computes Cn² and extracts the following variables into compressed NetCDF:

```
z, HGT, p, uvmet, wa, th, rh, PBLH, LANDMASK, slp,
T2, U10, V10, LH, HFX, UST, QKE, EL_PBL, TSQ
```

## Pipeline stages

Each simulation directory follows this four-stage layout:

```
sim_YYYY-MM-DD/
├── 1_forcing/     # CERRA grib files (pulled via rclone, deleted after WPS)
├── 2_wps/         # WPS output: met_em*.nc files (deleted after post-proc)
├── 3_wrf/         # WRF run directory and wrfout files
└── 4_postproc/    # cn2 NetCDF files (final output)
```

Intermediate data are garbage-collected automatically: CERRA files are removed once WPS completes; `met_em*.nc` files are removed once Cn² output exists.

## Machine configurations

Select the active machine by symlinking `env.yaml` to one of the provided environment files:

```bash
ln -sf env_<machine>.yaml env.yaml
```

| Machine | File | Pipeline | Purpose |
|---|---|---|---|
| `turbulence` | `env_turbulence.yaml` | `p_preproc` | WPS preprocessing (SSD tmp dir) |
| `chaos` | `env_chaos.yaml` | `p_default` | Full pipeline |
| `snellius` | `env_snellius.yaml` | `p_snellius` | WRF + post-proc on HPC (32 cores, 12 h walltime) |

### Typical two-machine workflow

1. **Turbulence** — run `p_preproc` to pull CERRA and produce `met_em*.nc` files.
2. **Snellius** — run `p_snellius` to execute WRF (32 MPI tasks) and post-processing on scratch, then copy results back.

## Running the pipeline

The entry point is `cli.py`, which reads `env.yaml` and exposes the selected pipeline:

```bash
# Run a single simulation
python cli.py run --stages=<stage> sim_YYYY-MM-DD

# Submit a SLURM array job (reads sim dirs from .array_sim_dirs)
sbatch --array=1-N slurm_<machine>.job.sh <stage>
```

`wrf_massive` tracks completion via `.done` marker files; stages are skipped if already finished.