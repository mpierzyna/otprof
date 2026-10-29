"""Step 1: download ERA5 for the times and area of `params.yaml` → `data/era5.nc`.

ERA5 comes from the public ARCO-ERA5 store on Google Cloud (no account needed). The store keeps every time step for
the whole globe in one piece, so each time step downloads about 1 GB, whatever the area. Delete the file to download
again.
"""

import pathlib

import pandas as pd
import yaml

from otprof.sources import Area, era5

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    params = yaml.safe_load((HERE / "params.yaml").read_text())
    out_file = HERE / "data" / "era5.nc"
    if out_file.exists():
        raise SystemExit(f"{out_file} exists; delete it to download again.")
    out_file.parent.mkdir(exist_ok=True)

    times = pd.date_range(params["start"], params["end"], freq=params["freq"])
    ds = era5.fetch(times, Area(**params["area"]))
    print(f"{len(times)} time steps, {ds.sizes['latitude']} x {ds.sizes['longitude']} grid points")
    print(f"{era5.get_download_size(times)['download'] / 1e9:.1f} GB to download, {ds.nbytes / 1e6:.1f} MB on disk")
    ds.to_netcdf(out_file)
    print(f"Wrote {out_file}")
