"""Pipeline for 1-year WRF run over NL"""

import datetime
import logging
import pathlib
import random
import string

from wrf_massive.base import Pipeline, Resources, Simulation, Stage
from wrf_massive.config import yaml_to_dict
from wrf_massive.stages.forcing import PullCerraStage
from wrf_massive.stages.misc import GarbageCollectStage, MarkDone, StageArray
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage
from wrf_massive.stages.wps import WPSStage
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
    # note: model_dump() now carries `tmp_work_root`, so override the key instead of passing it twice
    _wps_tmp_ssd = WPSStage(
        **{**_wps.model_dump(), "tmp_work_root": "/media/ssd_4tb_qvo/wrf_massive_tmp"},
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

if env["machine"] == "delftblue":
    # _wps_tmp_ssd = WPSStage(
    #     **{**_wps.model_dump(), "tmp_work_root": "/tmp/wrf-massive"},  # node-local SSD
    # )
    p_preproc = Pipeline(
        cerra=_update_resources(
            _cerra,
            n_tasks=1,
            cpus_per_task=8,
        ),
        wps=_update_resources(
            # _wps_tmp_ssd,
            _wps,
            n_tasks=4,
            cpus_per_task=1,
            walltime=datetime.timedelta(hours=1),
        ),
    )

if env["machine"] == "snellius":
    # For Snellius: run WRF and postproc with more ressources.
    # Minimum alloc: 16 cores, 28 GB RAM
    # At 32 cores, 0.59s per 10s step -> ca 16x real-time -> 5.5 sim days in 8.25h -> 9.5h with buffer
    # Update: took ~9h, so set to 12h with buffer for copying and memory bandwith saturation.
    # tmp-dir behaviour is configured per substage now. Copies, so the shared `_wrf`/`_cn2`
    # instances used by `p_default` keep their defaults.
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
            # "tmp_skip_teardown": True,  # keep on scratch for debugging
        }
    )
    _cn2_tmp = _cn2.model_copy(
        update={
            "tmp_work_root": TMP_ROOT,
            # "tmp_skip_teardown": True,  # keep on scratch for debugging
        }
    )

    # Pipeline in single slurm job
    # p_snellius = Pipeline(
    #     wrf_cn2=StageArray(
    #         stages={
    #             "wrf": _wrf_tmp,
    #             "cn2": _cn2_tmp,
    #             # "sim_done": _sim_done,  # work_dir is the sim dir -> skipped for tmp with a warning
    #         },
    #         tmp_work_root=TMP_ROOT,
    #         resources=Resources(
    #             n_tasks=32,
    #             cpus_per_task=1,
    #             mem_per_cpu="2000M",
    #             walltime=datetime.timedelta(hours=12),
    #         ),
    #     ),
    # )

    # Pipeline in separate slurm jobs
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


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    sim = Simulation.from_disk("sim_2017-01-01")
    p_dev = Pipeline(cn2=_cn2)
    p_dev.run(sim, force_run=True)
    # print(p["cn2"].get_inputs(s))
    # p["cn2"].run_single(s, 0)

    # p.run("cerra")
    # p.run("cerra", force=True)
    # p.run("wps")
    # p.run("wrf", force=True)
    # p.run("cn2", force=True)
