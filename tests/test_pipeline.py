import numpy as np
import pydantic
import pytest
import xarray as xr

from otprof.pipeline import BaseDataPipeline
from otprof.transforms import DatasetScaler, QuantileMapper, Transforms

TIMES = np.array(
    ["2019-01-01T00", "2019-07-01T12", "2020-01-01T00", "2018-01-01T00", "2018-07-01T12", "2017-01-01T00"],
    "datetime64[ns]",
)


class SynthPipeline(BaseDataPipeline):
    def load_data(self) -> None:
        rng = np.random.default_rng(0)
        coords = {"time": ("sample", TIMES), "lsm": ("sample", [1.0, 0.0, 1.0, 1.0, 0.0, 1.0])}
        self.ds_X = xr.Dataset({"a": (("sample", "level"), rng.normal(size=(6, 3)))}, coords=coords)
        self.ds_y = xr.Dataset({"b": (("sample", "level"), rng.lognormal(size=(6, 3)))}, coords=coords)


def land_train(p):
    return p.ds_X["time"].dt.year.isin([2019, 2020]) & (p.ds_X["lsm"] > 0.5)


def double_a(ds_X, ds_y):
    return {"a2": 2 * ds_X["a"]}, {}


def _pipeline(**kw) -> SynthPipeline:
    config = dict(
        sel_train={"year": [2019, 2020]},
        sel_val={"year": [2018]},
        sel_test={"year": [2017]},
        vars_X=["a"],
        vars_y=["b"],
        pre_X=[DatasetScaler({"a": (None, "standard", None)})],
        pre_y=[Transforms({"b": ["log10_clip", {"min": 1e-8}]}), DatasetScaler({"b": (None, "minmax", None)})],
    )
    return SynthPipeline(**{**config, **kw})


def test_time_selection_ands_components():
    dp = _pipeline(sel_val={"year": [2018, 2019], "hour": [12]}).setup()
    np.testing.assert_array_equal(dp.select("val"), [False, True, False, False, True, False])
    np.testing.assert_array_equal(dp.select("train"), [True, True, True, False, False, False])


def test_function_selection():
    dp = _pipeline(sel_train=land_train).setup()
    np.testing.assert_array_equal(dp.select("train"), [True, False, True, False, False, False])


def test_lambdas_rejected():
    with pytest.raises(pydantic.ValidationError, match="named function"):
        _pipeline(sel_test=lambda p: p.ds_X["lsm"] > 0)
    dp = _pipeline()
    with pytest.raises(pydantic.ValidationError, match="named function"):
        dp.sel_test = lambda p: p.ds_X["lsm"] > 0


def test_fit_on_train_split_after_transforms():
    dp = _pipeline().setup()
    _, ds_y = dp.split("train", transform=False)
    np.testing.assert_allclose(dp.pre_y[1].loc_["b"], np.log10(ds_y["b"]).min())
    _, ds_y_t = dp.split("train", transform=True)
    assert float(ds_y_t["b"].min()) == 0 and float(ds_y_t["b"].max()) == 1


def test_backward_inverts_forward_then_post():
    dp = _pipeline().setup()
    _, ds_y = dp.split("test", transform=False)
    xr.testing.assert_allclose(dp.backward_y(dp.forward_y(ds_y)), ds_y)

    shift = ds_y.copy(deep=True)
    shift["b"] = shift["b"] + 1  # "reference" one higher: the mapping shifts every value by +1
    dp.post_y = [QuantileMapper(n_quantiles=5, vars=["b"], grid="sine").fit(X_obs=shift, X_model=ds_y)]
    xr.testing.assert_allclose(dp.backward_y(dp.forward_y(ds_y)), shift)
    xr.testing.assert_allclose(dp.backward_y(dp.forward_y(ds_y), post=False), ds_y)


def test_roundtrip_without_data_and_lazy_functions(tmp_path):
    dp = _pipeline(sel_train=land_train, fe_fns=[double_a], vars_X=["a", "a2"]).setup()
    dp.post_y = [QuantileMapper(n_quantiles=3, vars=["b"]).fit(X_obs=dp.ds_y, X_model=dp.ds_y)]
    dp.to_disk(tmp_path)

    # Loads without the functions (e.g. inference): only using them fails, naming what is missing
    loaded = SynthPipeline.from_disk(tmp_path)
    assert loaded.ds_X is None and loaded.is_fitted
    assert loaded.sel_train == "land_train" and loaded.fe_fns == ["double_a"]
    loaded.load_data()
    with pytest.raises(ValueError, match="land_train"):
        loaded.select("train")

    loaded = SynthPipeline.from_disk(tmp_path, functions={"land_train": land_train, "double_a": double_a})
    loaded.setup()  # steps already fitted: not refitted
    for stage in ["train", "val", "test"]:
        for a, b in zip(dp.split(stage, transform=True), loaded.split(stage, transform=True)):
            xr.testing.assert_identical(a, b)
    _, y = dp.split("test", transform=True)
    xr.testing.assert_identical(dp.backward_y(y), loaded.backward_y(y))


def test_from_disk_unknown_class(tmp_path):
    _pipeline().to_disk(tmp_path)
    with pytest.raises(ValueError, match="SynthPipeline"):
        BaseDataPipeline.from_disk(tmp_path)
    assert isinstance(BaseDataPipeline.from_disk(tmp_path, pipelines={"SynthPipeline": SynthPipeline}), SynthPipeline)
