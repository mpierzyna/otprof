"""Pipeline for 1-year WRF run over NL"""

import logging
import pathlib
import random
import string
import datetime

from simulations import sim_dev
from wrf_massive.base import Pipeline, Resources, Stage, Simulation
from wrf_massive.config import yaml_to_dict
from wrf_massive.stages.forcing import PullCerraStage
from wrf_massive.stages.misc import GarbageCollectStage, StageArray, MarkDone
from wrf_massive.stages.postproc import PostprocCn2Stage
from wrf_massive.stages.wps import WPSStage, WPSTmpDirStage
from wrf_massive.stages.wrf import WRFStage


def get_random_id(n: int = 6) -> str:
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(n))


def _update_resources(stage: Stage, **resources) -> Stage:
    """Helper to update n_tasks of a stage. Returns deep copy."""
    import copy

    stage = copy.deepcopy(stage)
    stage.resources = stage.resources.model_copy(update=resources)
    return stage


# Load host-specific environment settings
env = yaml_to_dict(pathlib.Path("env.yaml").read_text())

# Setup stages
_cerra = PullCerraStage(
    work_dir="1_forcing",
    remote_path="tudelft:staff-umbrella/HBaki/CERRA",
    remote_flist_path="CERRA_files.txt.gz",
    n_transfers=4,
    resources=Resources(n_tasks=1, cpus_per_task=4, mem_per_cpu="1G"),
)

_wps = WPSStage(
    work_dir="2_wps",
    forcing_dir=_cerra.work_dir,  # 1_forcing
    namelist_tmpl_path="namelist.tmpl.wps",
    **env["wps"],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),  # serial WPS
)

# forcing data no longer needed after WPS finished -> Clear space.
_cerra_gc = GarbageCollectStage(
    work_dir=_cerra.work_dir,  # 1_forcing
    glob_pattern="*/*",
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
    resources=Resources(n_tasks=4, cpus_per_task=1, mem_per_cpu="1G"),
)
_cn2 = PostprocCn2Stage(
    work_dir="4_postproc",
    wrfout_dir=_wrf.work_dir,  # 3_wrf
    domain=1,
    extract_vars=[
        "z",
        "HGT",
        "p",
        "uvmet",
        "wa",
        "th",  # potential temperature
        # "tk",
        "rh",
        "PBLH",
        "LANDMASK",
        # # "QRAIN",
        # # "dbz",
        "slp",
        "T2",
        # "TH2",
        "U10",
        "V10",
        "LH",
        "HFX",
        "UST",
        # "ZNT",
        # "Z0",
        "QKE",
        ("EL_PBL", "bottom_top_stag"),
        "TSQ",
    ],
    compression=False,
    run_parallel=True,
    resources=Resources(n_tasks=1, cpus_per_task=8, mem_per_cpu="1GB"),
)


def cn2_files_exist(s: Simulation) -> bool:
    """Check if any CN2 output files exist for simulation s."""
    cn2_dir = _cn2.get_work_dir(s)
    return len(list(pathlib.Path(cn2_dir).glob("wrfout*cn2.nc"))) > 0


_wps_gc = GarbageCollectStage(
    work_dir=_wps.work_dir,  # 2_wps
    glob_pattern="met_em*.nc",
    armed=True,
    run_cond_fn=cn2_files_exist,
)
# mark whole simulation dir as done when all stages complete
_sim_done = MarkDone(work_dir=".", run_cond_fn=cn2_files_exist)

# Assemble pipeline
p_default = Pipeline(
    cerra=_cerra,
    wps=_wps,
    cerra_gc=_cerra_gc,
    wrf=_wrf,
    cn2=_cn2,
    sim_done=_sim_done,
)


if env["machine"] == "turbulence":
    # For turbulence: preprocessing-only pipeline with tmp dir on SSD
    # Also, artifically increase number of required tasks to avoid filling up filesystem with CERRA and WPS.
    # Turbulence has 48 cores, so requesting 8 cores per job will limit to max. 6 concurrent WPS jobs.
    _wps_tmp_ssd = WPSTmpDirStage(
        **_wps.model_dump(),
        tmp_dir_root="/media/ssd_4tb_qvo/wrf_massive_tmp",
    )
    p_preproc = Pipeline(
        cerra=_update_resources(
            _cerra,
            n_tasks=1,
            cpus_per_task=8,
        ),
        wps=_update_resources(
            _wps_tmp_ssd,
            n_tasks=1,
            cpus_per_task=8,
        ),
        # cerra_gc=_update_resources(_cerra_gc, n_tasks=8),  # should just fit. 56 * (16GB CERRA + 1 GB WPS) = 952 GB
    )

if env["machine"] == "snellius":
    # For Snellius: run WRF and postproc with more ressources.
    # Minimum alloc: 16 cores, 28 GB RAM
    # At 32 cores, 0.59s per 10s step -> ca 16x real-time -> 5.5 sim days in 8.25h -> 9.5h with buffer
    # Update: took ~9h, so set to 12h with buffer for copying and memory bandwith saturation.
    p_snellius = Pipeline(
        wrf_cn2=StageArray(
            stages={"wrf": _wrf, "cn2": _cn2},
            tmp_work_root="/scratch-shared/mpierzyna/",
            resources=Resources(
                n_tasks=32,
                cpus_per_task=1,
                mem_per_cpu="1500M",
                walltime=datetime.timedelta(hours=12),
            ),
            stage_tmp_teardown_globs={
                "wrf": [
                    "namelist.input",
                    "myoutfields.txt",
                    ".gitignore",
                ],  # move only settings back
            },
        ),
        # will be submitted as separate job because otherwise WPS first gets moved to scratch. Not ideal.
        wps_gc=StageArray(
            stages={
                "wps_gc": _wps_gc,
                "sim_done": _sim_done,
            },
            resources=Resources(
                n_tasks=1,
                cpus_per_task=16,
                mem_per_cpu="1G",
                walltime=datetime.timedelta(minutes=10),
            ),
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    p_default.run(sim_dev)
    # print(p["cn2"].get_inputs(s))
    # p["cn2"].run_single(s, 0)

    # p.run("cerra")
    # p.run("cerra", force=True)
    # p.run("wps")
    # p.run("wrf", force=True)
    # p.run("cn2", force=True)
