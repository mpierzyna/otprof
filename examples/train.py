"""Train a model from an experiment config.

    uv run python examples/train.py examples/configs/wrf_pl_h.yaml --data-root <domain data dir>

Loads the dataset files of the domain, fits the scalers on the training split and trains. The run directory
(`<save-dir>/lightning_logs/<version>/`) holds `config.yaml`, `data_pipeline/` (fitted scalers as JSON), the
TensorBoard logs (`tensorboard --logdir lightning_logs`) and the 5 best checkpoints by validation loss.
"""

import argparse
import pathlib

import torch

from otprof.experiment import Experiment
from otprof.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--data-root", type=pathlib.Path, help="domain data dir (default: the config's data.root)")
    parser.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--version", help="run name under lightning_logs/ (default: version_<N>)")
    parser.add_argument("--resume-from", help="checkpoint to continue from")
    parser.add_argument("--devices", default="auto", help="'auto' or GPU ids, e.g. 0,1")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    devices = args.devices if args.devices == "auto" else [int(d) for d in args.devices.split(",")]
    exp = Experiment.from_config(args.config, data_root=args.data_root).setup()
    run_dir = train(exp, save_dir=args.save_dir, version=args.version, resume_from=args.resume_from, devices=devices)
    print(f"Run saved to {run_dir}")
