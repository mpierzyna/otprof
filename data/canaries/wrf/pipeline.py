"""Pipeline for a single 5-day WRF run over the Canary Islands.

Forcing is pulled directly from the Copernicus CDS (CERRA + ERA5), so no local mirror is needed. All stages
run locally on a single machine (chaos).
"""

import logging
import pathlib

from simulations import sim_canaries
from wrf_massive.base import Pipeline, Resources, Simulation
from wrf_massive.config import yaml_to_dict
from wrf_massive.stages.forcing import CdsRequestSpec, PullCdsStage, PullCerraStage
from wrf_massive.stages.forcing.variables import (
    CERRA_PRESSURE_LEVEL_VARIABLES,
    CERRA_PRESSURE_LEVELS,
    CERRA_SINGLE_LEVEL_VARIABLES,
)
from wrf_massive.stages.misc import GarbageCollectStage, MarkDone
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage
from wrf_massive.stages.wps import WPSStage
from wrf_massive.stages.wrf import WRFStage

# Load host-specific environment settings
env = yaml_to_dict(pathlib.Path("env.yaml").read_text())

# ERA5 supplies the fields CERRA lacks for WRF: the 4-layer soil state, and 10m wind as u/v components
# (CERRA archives 10m wind only as speed/direction). metgrid merges these with CERRA (fg_name='ERA5','CERRA').
ERA5_GAPFILL_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]

# CERRA reanalysis (analysis product) from CDS. Its projected 5.5km grid does not support geographic
# subsetting, so the full native domain is downloaded (use_area=False) and cropped by WPS.
_cerra_cds = PullCdsStage(
    work_dir="1_forcing",
    prefix="CERRA",
    requests=[
        CdsRequestSpec(
            dataset="reanalysis-cerra-pressure-levels",
            variables=CERRA_PRESSURE_LEVEL_VARIABLES,
            levels=CERRA_PRESSURE_LEVELS,
            file_suffix="PRES",
            product_type=["analysis"],
            use_area=False,
            extra_params={"data_type": ["reanalysis"]},
        ),
        CdsRequestSpec(
            dataset="reanalysis-cerra-single-levels",
            variables=CERRA_SINGLE_LEVEL_VARIABLES,
            file_suffix="SFC",
            product_type=["analysis"],
            use_area=False,
            extra_params={"data_type": ["reanalysis"], "level_type": ["surface_or_atmosphere"]},
        ),
    ],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="4G"),
)

# ERA5 gap-filler (soil + 10m wind). ERA5 supports lat/lon cropping to Simulation.area, validated here
# against the WRF domain derived from namelist.tmpl.wps.
_era5_cds = PullCdsStage(
    work_dir="1_forcing",
    prefix="ERA5",
    namelist_wps_path="namelist.tmpl.wps",
    requests=[
        CdsRequestSpec(
            dataset="reanalysis-era5-single-levels",
            variables=ERA5_GAPFILL_VARIABLES,
            file_suffix="SFC",
        ),
    ],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
)

_cerra = PullCerraStage(
    work_dir="1_forcing",
    remote_path="tudelft:staff-umbrella/HBaki/CERRA",
    remote_flist_path="CERRA_files.txt.gz",
    n_transfers=4,
    resources=Resources(n_tasks=1, cpus_per_task=4, mem_per_cpu="1G"),
)

_wps = WPSStage(
    work_dir="2_wps",
    forcing_dir=_cerra_cds.work_dir,  # 1_forcing
    namelist_tmpl_path="namelist.tmpl.wps",
    **env["wps"],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="16G"),  # serial WPS
)

# Forcing data no longer needed after WPS finished -> clear space.
_forcing_gc = GarbageCollectStage(
    work_dir=_cerra_cds.work_dir,  # 1_forcing
    glob_pattern="*.grb",
    armed=True,
    run_cond_fn=_wps.is_done,  # double-check WPS completion
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
)

_wrf = WRFStage(
    work_dir="3_wrf",
    met_em_dir=_wps.work_dir,  # 2_wps
    namelist_tmpl_path="namelist.tmpl.input",
    myoutfields_path="myoutfields.txt",
    **env["wrf"],
    resources=Resources(n_tasks=48, cpus_per_task=1, mem_per_cpu="1G"),  # chaos: 128 cores available
)

_cn2 = Cn2PostProcStage(
    work_dir="4_postproc",
    wrfout_dir=_wrf.work_dir,  # 3_wrf
    domain=1,
    compression=True,
    run_parallel=True,
    resources=Resources(n_tasks=1, cpus_per_task=8, mem_per_cpu="1GB"),
)


def cn2_files_exist(s: Simulation) -> bool:
    """Check if any CN2 output files exist for simulation s."""
    cn2_dir = _cn2.get_work_dir(s)
    return len(list(pathlib.Path(cn2_dir).glob("wrfout*cn2.nc"))) > 0


# met_em*.nc no longer needed once Cn2 output exists -> clear space.
_wps_gc = GarbageCollectStage(
    work_dir=_wps.work_dir,  # 2_wps
    glob_pattern="met_em*.nc",
    armed=True,
    run_cond_fn=cn2_files_exist,
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
)

# Mark whole simulation dir as done when all stages complete.
_sim_done = MarkDone(
    work_dir=".",
    run_cond_fn=cn2_files_exist,
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
)

# Assemble pipeline
p_cds = Pipeline(
    cerra=_cerra_cds,
    era5=_era5_cds,
    wps=_wps,
    forcing_gc=_forcing_gc,
    wrf=_wrf,
    cn2=_cn2,
    wps_gc=_wps_gc,
    sim_done=_sim_done,
)

p_local = Pipeline(
    cerra=_cerra,
    wps=_wps,
    forcing_gc=_forcing_gc,
    wrf=_wrf,
    cn2=_cn2,
    wps_gc=_wps_gc,
    sim_done=_sim_done,
)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    p_cds.run(sim_canaries)
