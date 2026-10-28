"""Step 1: download the GFS forecast of `params.yaml` at the locations → `data/gfs.nc`.

GFS comes from the NOAA archive on AWS (no account needed). Only the needed fields are downloaded (a few MB per lead
time), from the grid points nearest to the locations. Delete the file to download again.
"""

import pathlib

import xarray as xr
import yaml

from otprof.sources import gfs

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    params = yaml.safe_load((HERE / "params.yaml").read_text())
    out_file = HERE / "data" / "gfs.nc"
    if out_file.exists():
        raise SystemExit(f"{out_file} exists; delete it to download again.")
    out_file.parent.mkdir(exist_ok=True)

    names = list(params["locations"])
    lat, lon = zip(*params["locations"].values())
    leads = []
    for lead in params["leads"]:
        leads.append(gfs.fetch(params["cycle"], lead, lat, lon))
        print(f"+{lead} h downloaded")
    ds = xr.concat(leads, "time").assign_coords(name=("loc", names))
    ds.to_netcdf(out_file)
    print(f"Wrote {out_file}")
