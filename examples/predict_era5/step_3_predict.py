"""Step 3: predict Cn2 profiles from the inputs → `data/y.nc`.

Downloads the published model from Hugging Face on first use (cached afterwards) and predicts at its default heights
above ground (the 99 WRF levels it was trained on). Output per time and grid point and height (dim bottom_top):
`lcn2` (log10 Cn2 [m-2/3]), the WRF turbulence quantities `QKE`, `TSQ`, `EL_PBL`, the heights `z_agl` [m], and per
profile the Fried parameter `r0` [cm] and the log10 scintillation index `lsi2` (1550 nm, plane wave).
"""

import pathlib

import xarray as xr
import yaml

from otprof.models import sqf_h
from otprof.predict import load_model, predict

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    params = yaml.safe_load((HERE / "params.yaml").read_text())
    if params["model"] is None:
        predictor = load_model(sqf_h.HF_REPO_ID, revision=sqf_h.HF_REVISION)
    else:
        predictor = load_model(params["model"])

    X = xr.load_dataset(HERE / "data" / "X.nc")
    y = predict(predictor, X)
    y.to_netcdf(HERE / "data" / "y.nc")
    print(y)
    print(f"Wrote {HERE / 'data' / 'y.nc'}")
