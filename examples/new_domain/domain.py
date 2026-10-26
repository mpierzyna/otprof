"""Shared by the step scripts: data directory and parameters of a domain (`params.yaml`)."""

import argparse
import os
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
# Parent of the per-domain data directories; set OTPROF_DATA_ROOT to write elsewhere (e.g. to compare two runs)
DATA_ROOT = pathlib.Path(os.environ.get("OTPROF_DATA_ROOT", HERE / "data"))


def parse(description: str | None = None) -> tuple[pathlib.Path, dict]:
    """Domain name from the command line → (data directory, parameters from `params.yaml`)."""
    params = yaml.safe_load((HERE / "params.yaml").read_text())["domains"]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("domain", choices=sorted(params))
    name = parser.parse_args().domain
    root = DATA_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    return root, params[name]
