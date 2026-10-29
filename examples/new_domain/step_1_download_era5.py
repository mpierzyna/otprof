"""Step 1: download ERA5 for the time steps of the WRF output → `data/<domain>/1_raw/era5.nc`.

Every time step is downloaded for the whole globe and cut to the area, so this takes a while. Delete the file to
download again.
"""

from otprof import dataset
from otprof.sources import Area, era5

import domain

if __name__ == "__main__":
    root, params = domain.parse(__doc__)
    out_file = root / "1_raw" / "era5.nc"
    if out_file.exists():
        raise SystemExit(f"{out_file} exists; delete it to download again.")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    wrf_files = sorted((domain.HERE / params["wrf_output"]).glob("**/wrfout_aux_*_cn2.nc"))
    times = dataset.get_wrf_times(wrf_files)
    print(f"{len(wrf_files)} WRF files, {len(times)} time steps")

    ds = era5.fetch(times, Area(**params["area"]))
    print(ds)
    print(f"{era5.get_download_size(times)['download'] / 1e9:.1f} GB to download, {ds.nbytes / 1e9:.1f} GB on disk")
    ds.to_netcdf(out_file)
    print(f"Wrote {out_file}")
