"""Pipeline for a single 5-day WRF run over the Canary Islands.

Forcing comes either directly from the Copernicus CDS (CERRA + ERA5, `p_cds`) or from the TU Delft CERRA mirror
(`p_default`). On HPC, WPS pre-processing runs on DelftBlue (`p_preproc`) and WRF + post-processing on Snellius
(`p_snellius`); the active pipeline is selected via `env.yaml`.
"""

import datetime
import pathlib

from wrf_massive.base import Pipeline, Resources, Simulation, Stage
from wrf_massive.config import yaml_to_dict
from wrf_massive.stages.forcing import PullCerraStage
from wrf_massive.stages.misc import GarbageCollectStage, MarkDone
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage
from wrf_massive.stages.wps import WPSStage
from wrf_massive.stages.wrf import WRFStage


def _update_resources(stage: Stage, **resources) -> Stage:
    """Helper to update n_tasks of a stage. Returns deep copy."""
    import copy

    stage = copy.deepcopy(stage)
    stage.resources = stage.resources.model_copy(update=resources)
    return stage


# Load host-specific environment settings
env = yaml_to_dict(pathlib.Path("env.yaml").read_text())


# Preproc
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
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="16G"),  # serial WPS
)

# Forcing data no longer needed after WPS finished -> clear space.
_forcing_gc = GarbageCollectStage(
    work_dir=_cerra.work_dir,  # 1_forcing
    glob_pattern="**/*.grb",  # CDS: flat, mirror: in year subdirs
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


if env["machine"] == "delftblue":
    # WPS pre-processing with forcing from the TU Delft CERRA mirror. WPS runs dmpar on 4 tasks.
    p_preproc = Pipeline(
        cerra=_update_resources(
            _cerra,
            n_tasks=1,
            cpus_per_task=8,
        ),
        wps=_update_resources(
            _wps,
            n_tasks=4,
            cpus_per_task=1,
            mem_per_cpu="3G",
            walltime=datetime.timedelta(hours=1),
        ),
    )

if env["machine"] == "snellius":
    # For Snellius: run WRF and postproc with more ressources.
    # Domain size is comparable to the NL campaign (200x125 vs. 144x192), where 32 cores took ~9h per 5.5 sim days.
    # tmp-dir behaviour is configured per substage. Copies, so the shared `_wrf`/`_cn2` instances used by
    # `p_default` keep their defaults.
    TMP_ROOT = pathlib.Path("/scratch-shared/mpierzyna/")
    _wrf_tmp = _wrf.model_copy(
        update={
            "tmp_work_root": TMP_ROOT,
            "tmp_teardown_globs": [  # move only settings and scripts back
                "setup_wrf.sh",
                "run_wrf.sh",
                "namelist.input",
                "myoutfields.txt",
                ".gitignore",
            ],
            "tmp_skip_teardown": True,  # leave symlink in place so Cn2 can postproc
        }
    )
    _cn2_tmp = _cn2.model_copy(
        update={
            "tmp_work_root": TMP_ROOT,
            "tmp_skip_teardown": True,
        }
    )

    p_snellius = Pipeline(
        wrf=_update_resources(
            _wrf_tmp,
            n_tasks=32,
            cpus_per_task=1,
            mem_per_cpu="1000M",
            walltime=datetime.timedelta(hours=12),
        ),
        cn2=_update_resources(
            _cn2_tmp,
            n_tasks=1,
            cpus_per_task=16,
            mem_per_cpu="2000M",
            walltime=datetime.timedelta(minutes=15),
        ),
    )
