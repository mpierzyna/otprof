import json

import numpy as np
import pytest
import xarray as xr

from otprof.transforms import STEPS, DatasetScaler, QuantileMapper, Transforms, step_from_dict


def _ds(seed: int = 0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "a": (("sample", "level"), rng.normal(5, 2, (200, 4)).astype("float32")),
            "b": (("sample",), rng.lognormal(0, 1, 200)),  # float64
        },
        coords={"level": [10, 20, 30, 40]},
    )


def _roundtrip(step):
    return step_from_dict(json.loads(json.dumps(step.to_dict())))


@pytest.mark.parametrize(
    "config",
    [
        {"a": (None, "standard", None), "b": (None, "minmax", None)},
        {"a": (["sample"], "quantile", (0.01, 0.99)), "b": (None, "robust", (0.25, 0.75))},
    ],
)
def test_scaler_roundtrip_exact(config):
    ds = _ds()
    scaler = DatasetScaler(config).fit(ds)
    loaded = _roundtrip(scaler)
    for v in ds:
        xr.testing.assert_identical(scaler.loc_[v], loaded.loc_[v])
        xr.testing.assert_identical(scaler.scale_[v], loaded.scale_[v])
    xr.testing.assert_identical(scaler.transform(ds), loaded.transform(ds))
    xr.testing.assert_identical(scaler.inverse_transform(ds), loaded.inverse_transform(ds))


def test_unfitted_scaler_roundtrip():
    loaded = _roundtrip(DatasetScaler({"a": (None, "standard", None)}))
    assert not loaded.is_fitted and loaded.config == {"a": (None, "standard", None)}


@pytest.mark.parametrize("grid", ["linear", "sine"])
@pytest.mark.parametrize("per_level", [False, True])
def test_quantile_mapper_roundtrip_exact(grid, per_level):
    obs, mod = _ds(0), _ds(1)
    qm = QuantileMapper(n_quantiles=20, vars=["a"], per_level=per_level, grid=grid).fit(X_obs=obs, X_model=mod)
    loaded = _roundtrip(qm)
    assert loaded.per_level_vars_ == qm.per_level_vars_
    xr.testing.assert_identical(qm.transform(_ds(2)), loaded.transform(_ds(2)))


def test_transforms_forward_inverse():
    ds = _ds()
    tf = Transforms({"b": ["log10_clip", {"min": 1e-8}]})
    out = tf.transform(ds)
    np.testing.assert_array_equal(out["b"], np.log10(ds["b"].clip(min=1e-8)))
    xr.testing.assert_identical(out["a"], ds["a"])  # not configured: unchanged
    xr.testing.assert_allclose(tf.inverse_transform(out), ds)
    xr.testing.assert_identical(_roundtrip(tf).transform(ds), out)


def test_unknown_names_fail_with_known_list():
    with pytest.raises(ValueError, match="log10_clip"):
        Transforms({"b": ["log", {}]})
    with pytest.raises(ValueError, match="scaler"):
        step_from_dict({"step": "nope"})


def test_steps_cover_all_classes():
    assert Transforms({}).to_dict()["step"] in STEPS
    assert DatasetScaler({}).to_dict()["step"] in STEPS
    assert QuantileMapper(n_quantiles=3, vars=[]).to_dict()["step"] in STEPS
