"""Data pipeline: loads inputs (X) and targets (y), splits them and owns the processing chain to and from the model.

A pipeline is used for training and evaluation (X and y files, three required selectors). Inference does not load
data: it restores the pipeline without data (`from_disk`) and only uses its chain:

    y = pipeline.backward_y(model(pipeline.forward_X(X)))

The chain consists of three lists of steps (`otprof.transforms`), applied in order:

- `pre_X`: inputs -> model space
- `pre_y`: targets -> model space (training); inverted in reverse order for the model outputs
- `post_y`: outputs only, after `pre_y` is inverted (e.g. output quantile mapping); fitted on predictions, then appended

Functions in the config (named selectors, `fe_fns`) are stored by name and resolved through `functions` only when
used, so a saved pipeline loads without them (e.g. for inference).

`ProfilePipeline` reads the dataset files of a domain (see `examples/new_domain`). What a model makes of them (variables,
transforms, scalers, default split) is defined with the model, e.g. `otprof.models.sqf_h.data.get_pipeline`, whose
pipeline class also computes the model inputs from the canonical dataset files when they are loaded.
"""

from __future__ import annotations

import abc
import json
import logging
import pathlib
from typing import Any, Callable, Literal, Self

import pydantic
import xarray as xr

from otprof.transforms import STEPS, InvertibleStep, Step, step_from_dict

logger = logging.getLogger(__name__)

TStage = Literal["train", "val", "test"]

# Selection by time: every listed component of `time` must be one of the values, e.g. {"year": [2019, 2020]}.
TimeComponent = Literal["year", "month", "day", "hour", "dayofyear"]
TimeSel = dict[TimeComponent, list[int]]

# Named function (pipeline) -> boolean mask over `sample`, for anything beyond time, e.g. land profiles only.
SelFn = Callable[..., xr.DataArray]

# Feature engineering (prototyping): (ds_X, ds_y) -> (new X variables, new y variables). Features that prove useful
# move into the model's `make_inputs`, so that inference computes them too.
FeFn = Callable[[xr.Dataset, xr.Dataset], tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]]


def _fn_name(fn: str | Callable) -> str:
    return fn if isinstance(fn, str) else fn.__name__


class BaseDataPipeline(pydantic.BaseModel, abc.ABC):
    """Implement all steps to prepare dataset for training.
    Most importantly, implements how data is split, and the processing chain between data and model.
    """

    # Train/val/test selection: a time selection (plain data) or a named function, stored by its name.
    # A string is the name of a function to be resolved through `functions`.
    sel_train: TimeSel | SelFn | str
    sel_val: TimeSel | SelFn | str
    sel_test: TimeSel | SelFn | str

    # Variables passed to the model
    vars_X: list[str]
    vars_y: list[str]

    # Feature engineering functions (prototyping, see `FeFn`); strings are names resolved through `functions`
    fe_fns: list[FeFn | str] = []

    # Processing chain (see module docstring)
    pre_X: list[Step] = []
    pre_y: list[InvertibleStep] = []
    post_y: list[Step] = []

    # Name -> function for the selectors and `fe_fns` of a loaded pipeline. Not saved.
    functions: dict[str, Callable] = pydantic.Field(default_factory=dict, exclude=True)

    # Datasets are set up through methods. Don't pass them.
    ds_X: xr.Dataset | None = None
    ds_y: xr.Dataset | None = None

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @pydantic.field_validator("sel_train", "sel_val", "sel_test", "fe_fns")
    @classmethod
    def _no_lambdas(cls, v: Any) -> Any:
        """Functions are saved by name, so they must have one."""
        for fn in v if isinstance(v, list) else [v]:
            if callable(fn) and fn.__name__ == "<lambda>":
                raise ValueError("Use a named function (def ...) instead of a lambda: functions are saved by name.")
        return v

    @pydantic.field_serializer("sel_train", "sel_val", "sel_test")
    def _ser_sel(self, sel: TimeSel | SelFn | str) -> TimeSel | str:
        return sel if isinstance(sel, dict) else _fn_name(sel)

    @pydantic.field_serializer("fe_fns")
    def _ser_fe_fns(self, fe_fns: list[FeFn | str]) -> list[str]:
        return [_fn_name(fn) for fn in fe_fns]

    def _resolve(self, fn: Callable | str) -> Callable:
        if callable(fn):
            return fn
        try:
            return self.functions[fn]
        except KeyError:
            raise ValueError(
                f"Function '{fn}' is not known. Pass it to from_disk(functions=...). Known: {sorted(self.functions)}"
            ) from None

    @property
    def is_fitted(self) -> bool:
        """All steps of `pre_X` and `pre_y` fitted (steps without state count as fitted)."""
        return all(getattr(s, "is_fitted", True) for s in self.pre_X + self.pre_y)

    @property
    def is_setup(self) -> bool:
        """Check if all necessary steps have been run"""
        return self.ds_X is not None and self.ds_y is not None and self.is_fitted

    ## Data (training and evaluation)

    @abc.abstractmethod
    def load_data(self) -> None:
        """Implement how dataset is loaded from disk."""

    def fill_gaps(self) -> None:
        """Implement how missing values are filled. Return gap-filled datasets."""

    def apply_fe_fns(self) -> None:
        """Add the variables of the feature engineering functions (`fe_fns`) to ds_X and ds_y, then keep `vars_X` and
        `vars_y` only."""

        def _check_nan(fe_dict: dict[str, xr.DataArray], ds_name: str) -> None:
            for v, da in fe_dict.items():
                if da.isnull().any():
                    logger.warning(f"Feature engineering produced NaNs in {ds_name} variable '{v}'")

        if self.ds_X is None or self.ds_y is None:
            raise ValueError("Datasets not loaded. Cannot make features.")

        for fe_fn in self.fe_fns:
            fe_fn = self._resolve(fe_fn)
            logger.info(f"Applying feature engineering function: {fe_fn.__name__}")
            X_fe, y_fe = fe_fn(self.ds_X, self.ds_y)

            _check_nan(X_fe, "X")
            _check_nan(y_fe, "y")

            X_fe_vars = list(X_fe.keys())
            if X_fe_vars:
                logger.info(f"Adding X features: {X_fe_vars}.")
                self.ds_X = xr.merge([self.ds_X, xr.Dataset(X_fe)], compat="override")
                # Rebind (not in-place `+=`) and dedup so apply_fe_fns stays idempotent and never
                # mutates an aliased list: `model_copy(update={"vars_X": ...})` stores the passed
                # object without copying (would corrupt a module-level vars constant), and a loaded
                # config already lists these features.
                self.vars_X = self.vars_X + [v for v in X_fe_vars if v not in self.vars_X]

            y_fe_vars = list(y_fe.keys())
            if y_fe_vars:
                logger.info(f"Adding y features: {y_fe_vars}.")
                self.ds_y = xr.merge([self.ds_y, xr.Dataset(y_fe)], compat="override")
                self.vars_y = self.vars_y + [v for v in y_fe_vars if v not in self.vars_y]

        # Only select specified variables
        self.ds_X = self.ds_X[self.vars_X]
        self.ds_y = self.ds_y[self.vars_y]

    def fit(self, force: bool = False) -> None:
        """Fit the steps of `pre_X` and `pre_y` on the training split; each step on the output of the previous one."""
        fitted = [s for s in self.pre_X + self.pre_y if getattr(s, "is_fitted", False)]
        if fitted and not force:
            # OOD domains load the trained steps (via load_steps) instead of fitting on the
            # new domain. Skip rather than refit so loaded steps are never silently discarded.
            logger.info("Steps already fitted (e.g. loaded for OOD inference). Skipping fit.")
            return

        logger.info("Fitting the X and y steps on training data.")
        ds_X_train, ds_y_train = self.split(stage="train", transform=False)
        _fit_steps(self.pre_X, ds_X_train[self.vars_X])
        _fit_steps(self.pre_y, ds_y_train[self.vars_y])

    def prepare_data(self) -> Self:
        """Load the data, fill gaps and apply `fe_fns`, without fitting (e.g. to evaluate a trained model)."""
        self.load_data()
        self.fill_gaps()
        self.apply_fe_fns()
        return self

    def setup(self) -> Self:
        """Run all setup steps: `prepare_data`, then fit the steps (skipped if already fitted)."""
        logger.info(f"This is {self.__class__.__name__}. Starting setup.")
        self.prepare_data()
        self.fit()
        logger.info("Data pipeline setup complete.")

        return self

    def select(self, stage: TStage) -> xr.DataArray:
        """Boolean mask over `sample` for `stage`, from a time selection or a selector function."""
        sel = {"train": self.sel_train, "val": self.sel_val, "test": self.sel_test}[stage]
        if isinstance(sel, dict):
            return self.select_time(sel)
        return self._resolve(sel)(self)

    def select_time(self, sel: TimeSel) -> xr.DataArray:
        """Boolean mask over `sample`: `time` matches every component listed in `sel` (all samples if empty)."""
        t = self.ds_X["time"].dt
        mask = xr.ones_like(self.ds_X["time"], dtype=bool)
        for component, values in sel.items():
            mask = mask & getattr(t, component).isin(values)
        return mask

    def split(self, stage: TStage, transform: bool) -> tuple[xr.Dataset, xr.Dataset]:
        """Split low-res and high-res data into training, validation, test."""
        mask = self.select(stage)
        ds_X = self.ds_X.sel(sample=mask)
        ds_y = self.ds_y.sel(sample=mask)

        # Transform if needed (always all steps of pre_X and pre_y)
        if transform:
            ds_X, ds_y = self.forward_X(ds_X), self.forward_y(ds_y)

        return ds_X, ds_y

    ## Processing chain (training, evaluation and inference)

    def forward_X(self, ds_X: xr.Dataset) -> xr.Dataset:
        """Inputs -> model space."""
        for step in self.pre_X:
            ds_X = step.transform(ds_X)
        return ds_X

    def forward_y(self, ds_y: xr.Dataset) -> xr.Dataset:
        """Targets -> model space."""
        for step in self.pre_y:
            ds_y = step.transform(ds_y)
        return ds_y

    def backward_y(self, ds_y: xr.Dataset, post: bool = True) -> xr.Dataset:
        """Model outputs -> physical targets: `pre_y` inverted in reverse order, then `post_y` (if `post`)."""
        for step in reversed(self.pre_y):
            ds_y = step.inverse_transform(ds_y)
        if post:
            for step in self.post_y:
                ds_y = step.transform(ds_y)
        return ds_y

    ## Saving and loading (config and steps, never data)

    def to_disk(self, dp_dir: str | pathlib.Path) -> None:
        """Save config (`pipeline.json`) and steps incl. fitted state (`steps.json`), without data."""
        dp_dir = pathlib.Path(dp_dir)
        dp_dir.mkdir(parents=True, exist_ok=True)

        exclude = {"ds_X", "ds_y", "pre_X", "pre_y", "post_y", "functions"}
        config = {"type": self.__class__.__name__, **self.model_dump(mode="json", exclude=exclude)}
        (dp_dir / "pipeline.json").write_text(json.dumps(config, indent=2))

        steps = {k: [s.to_dict() for s in getattr(self, k)] for k in ("pre_X", "pre_y", "post_y")}
        (dp_dir / "steps.json").write_text(json.dumps(steps))

    def load_steps(self, dp_dir: str | pathlib.Path, steps: dict[str, type] | None = None) -> Self:
        """Take over the (fitted) steps of a saved pipeline, e.g. the training scalers for an OOD domain.

        Run before `setup()`; fitting is then skipped.
        """
        d = json.loads((pathlib.Path(dp_dir) / "steps.json").read_text())
        for k in ("pre_X", "pre_y", "post_y"):
            setattr(self, k, [step_from_dict(s, steps) for s in d[k]])
        return self

    @classmethod
    def from_disk(
        cls,
        dp_dir: str | pathlib.Path,
        pipelines: dict[str, type[BaseDataPipeline]] | None = None,
        functions: dict[str, Callable] | None = None,
        steps: dict[str, type] | None = None,
    ) -> Self:
        """Load config and steps of a saved pipeline, without data (run `setup()` or `load_data()` for that).

        `pipelines` maps the saved class name to the class (needed if it is not `cls`), `functions` the names of
        selectors and `fe_fns` to the functions, `steps` the step names to classes (default: `otprof.transforms.STEPS`).
        """
        dp_dir = pathlib.Path(dp_dir)
        config = json.loads((dp_dir / "pipeline.json").read_text())
        name = config.pop("type")
        known = {cls.__name__: cls, **(pipelines or {})}
        if name not in known:
            raise ValueError(f"Unknown pipeline '{name}'. Pass it to from_disk(pipelines=...). Known: {sorted(known)}")
        target_cls = known[name]
        if not issubclass(target_cls, cls):
            raise ValueError(f"Saved pipeline class {target_cls} is not a subclass of requested {cls}")

        dp = target_cls(**config, functions=functions or {})
        return dp.load_steps(dp_dir, steps=steps or STEPS)


class ProfilePipeline(BaseDataPipeline):
    """Inputs file (time, loc, bottom_top) + WRF targets of a domain -> samples, invalid profiles dropped.

    `inputs` and `targets` are paths relative to the domain directory `data_root`, in the layout of
    `examples/new_domain`: the canonical inputs (`otprof.schema`) `4_inputs_interp/wrf_pl.nc` (WRF on the ERA5
    pressure levels), `3_inputs/era5_pl.nc`, `3_inputs/wrf_native.nc` and the targets `3_targets/y_wrf.nc`. The model
    computes its inputs and target variables in `make_inputs` / `make_targets` (e.g. `SqfhPipeline`); here both are
    taken as they are. The targets lose their top level (the inputs lose theirs to the gradients: N-1 input levels
    from N). Profiles with NaN Cn2 (masked by `otprof.dataset.filter_cn2`) are dropped.
    """

    data_root: pathlib.Path
    inputs: str = "4_inputs_interp/wrf_pl.nc"
    targets: str = "3_targets/y_wrf.nc"

    def make_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        """Model inputs from the canonical inputs (`ds` itself here: the inputs file already holds them)."""
        return ds

    def make_targets(self, ds_y: xr.Dataset) -> xr.Dataset:
        """Additional target variables of the model (none here)."""
        return ds_y

    def get_input_order(self, features: list[str]) -> list[str]:
        """Order of `features` in the model's input tensor, without loading data (for export and loading runs)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not define the order of the model inputs")

    def load_data(self) -> None:
        ds_X = self.make_inputs(xr.load_dataset(self.data_root / self.inputs)).drop_vars("p_level", errors="ignore")
        ds_y = self.make_targets(xr.load_dataset(self.data_root / self.targets).isel(bottom_top=slice(0, -1)))

        # Samples: stacked time and location
        ds_X = ds_X.stack(sample=("time", "loc")).transpose("sample", ...)
        ds_y = ds_y.stack(sample=("time", "loc")).transpose("sample", ...)

        # Drop invalid Cn2 profiles
        invalid = ds_y["cn2"].isnull().any("bottom_top")
        self.ds_X, self.ds_y = ds_X.sel(sample=~invalid), ds_y.sel(sample=~invalid)


def _fit_steps(steps: list[Step], ds: xr.Dataset) -> None:
    """Fit each step that has a `fit` on the output of the steps before it."""
    for i, step in enumerate(steps):
        if hasattr(step, "fit"):
            step.fit(ds)
        if i < len(steps) - 1:
            ds = step.transform(ds)
