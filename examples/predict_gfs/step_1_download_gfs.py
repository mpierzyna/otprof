"""Step 1: download the GFS forecast of `params.yaml` at the locations → `data/gfs.nc`.

GFS comes from the NOAA archive on AWS (no account needed). Only the needed fields are downloaded (global fields,
about 120 MB per lead time), and the values at the grid points nearest to the locations are kept. Delete the file to
download again.
"""

import pathlib

import pandas as pd
import yaml

from otprof.sources import Points, gfs

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    params = yaml.safe_load((HERE / "params.yaml").read_text())
    out_file = HERE / "data" / "gfs.nc"
    if out_file.exists():
        raise SystemExit(f"{out_file} exists; delete it to download again.")
    out_file.parent.mkdir(exist_ok=True)

    cycle = pd.Timestamp(params["cycle"])
    times = [cycle + pd.Timedelta(hours=lead) for lead in params["leads"]]
    ds = gfs.fetch(cycle, times, Points.from_dict(params["locations"]))
    ds.to_netcdf(out_file)
    print(f"Wrote {out_file}")
