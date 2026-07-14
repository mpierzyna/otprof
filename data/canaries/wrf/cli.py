#!/usr/bin/env python3
import sys
import logging
import pathlib

sys.path.append("../../")

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

import pipeline
from wrf_massive.cli import get_pipeline_cli
from wrf_massive.config import yaml_to_dict

# Load host-specific environment settings
env = yaml_to_dict(pathlib.Path("env.yaml").read_text())
p = getattr(pipeline, env["pipeline"])
cli = get_pipeline_cli(p)

if __name__ == "__main__":
    cli()
