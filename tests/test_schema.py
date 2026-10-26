import pytest

from otprof import schema


def test_valid_dataset_passes(canonical):
    assert schema.validate(canonical) is canonical


def test_all_problems_listed(canonical):
    ds = canonical.drop_vars("ust")
    ds["slp"] = ds["slp"] * 100  # Pa instead of hPa
    ds["hfx"].attrs["units"] = "W/m2"
    ds["u10"] = ds["u"]  # profile instead of surface
    with pytest.raises(schema.SchemaError) as e:
        schema.validate(ds)
    msg = str(e.value)
    for expected in ["missing `ust`", "`slp` outside", "`hfx` must have units", "`u10` is a surface variable"]:
        assert expected in msg


def test_pressure_grid_levels_checked(canonical):
    with pytest.raises(schema.SchemaError, match="pressure levels"):
        schema.validate(canonical.isel(level=slice(None, None, -1)))  # top to bottom


def test_model_grid_any_levels(canonical):
    ds = canonical.isel(level=slice(0, 12)).drop_vars("level")
    schema.validate(ds, grid="model")
    with pytest.raises(schema.SchemaError):
        schema.validate(ds, grid="pressure")


def test_tk_or_th_required(canonical):
    with pytest.raises(schema.SchemaError, match="one of `tk`"):
        schema.validate(canonical.drop_vars("tk"))
    ds = canonical.rename(tk="th")
    ds["th"].attrs = {"units": "K"}
    schema.validate(ds)
