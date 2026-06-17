from __future__ import annotations

import abc
import importlib
import pathlib
from typing import List

import joblib
import pydantic
import lightning as L
import torch.utils.data
from typing_extensions import Self

from otprof.logging import get_logger, LogContext
from otprof.pipelines.base import BaseDataPipeline
from otprof.types import TStage
from research_tools.misc_tools.yaml_config import dict_to_yaml, yaml_to_dict

logger = get_logger()
logger_train = get_logger(context=LogContext.train)
logger_eval = get_logger(context=LogContext.eval)


class BaseExperiment(pydantic.BaseModel, abc.ABC):
    # Experiment settings
    name: str
    exp_dir: pathlib.Path | None = None

    # Data config
    dp: BaseDataPipeline
    features: List[str]  # features from X
    targets: List[str]  # targets from y
    forcings: List[str] = []  # forcing from y added to X

    # Training config
    batch_size: int
    max_epochs: int

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # Allow, e.g., numpy arrays or custom types
    )

    @pydantic.model_validator(mode="after")
    def validate_features_targets(self) -> Self:
        """Check that features and targets are in the data pipeline"""
        for f in self.features:
            if f not in self.dp.vars_X:
                raise ValueError(f"Feature {f} not found in data pipeline X variables.")
        for t in self.targets:
            if t not in self.dp.vars_y:
                raise ValueError(f"Target {t} not found in data pipeline y variables.")
        for f in self.forcings:
            if f not in self.dp.vars_y:
                raise ValueError(f"Forcing {f} not found in data pipeline y variables.")
        return self

    @pydantic.field_validator("dp")
    @classmethod
    def validate_dp_setup(cls, dp: BaseDataPipeline) -> BaseDataPipeline:
        """Check that data pipeline setup has been run"""
        if not dp.is_setup:
            dp.setup()
        return dp

    @abc.abstractmethod
    def get_torch_dataset(self, stage: TStage) -> torch.utils.data.Dataset:
        """Implement how to create a torch dataset for train/val/test"""

    @abc.abstractmethod
    def get_torch_loader(
        self,
        stage: TStage,
        data: torch.utils.data.Dataset | None = None,
    ) -> torch.utils.data.DataLoader:
        """Implement how torch data loaders for training, validation, testing are created.
        Optionally, preloaded dataset can be passed via `data` argument.
        """

    @abc.abstractmethod
    def get_lit_model(self, data_train: torch.utils.data.Dataset) -> L.LightningModule:
        """Implement how Lightning model is created"""

    def to_disk(self, exp_dir: str | pathlib.Path):
        logger.info(f"Saving experiment to {exp_dir}.")
        exp_dir = pathlib.Path(exp_dir)
        exp_dir.parent.mkdir(parents=True, exist_ok=True)
        self.exp_dir = exp_dir

        # Save data pipeline separately
        dp_path = exp_dir / "data_pipeline"
        self.dp.to_disk(dp_path)

        # Save experiment config
        self_dict = self.model_dump(exclude={"exp_dir": ..., "dp": ...})
        self_dict["dp_path"] = dp_path.name  # only relative path
        self_dict["dp_cls"] = f"{self.dp.__class__.__module__}:{self.dp.__class__.__qualname__}"

        # Save experiment class path so BaseExperiment.from_disk can instantiate the
        # original concrete subclass when loading via the base class method.
        self_dict["exp_cls"] = f"{self.__class__.__module__}:{self.__class__.__qualname__}"
        (exp_dir / "exp_config.yml").write_text(dict_to_yaml(self_dict))

    @classmethod
    def from_disk(cls, exp_dir: str | pathlib.Path) -> Self:
        logger.info(f"Loading experiment from {exp_dir}.")
        exp_dir = pathlib.Path(exp_dir)

        # Read config from disk
        exp_config = yaml_to_dict((exp_dir / "exp_config.yml").read_text())

        # Determine the target experiment class to instantiate
        target_cls = cls
        exp_cls_str = exp_config.get("exp_cls", "")
        if exp_cls_str:
            try:
                module_name, qualname = exp_cls_str.split(":", 1)
                module = importlib.import_module(module_name)
                obj = module
                for attr in qualname.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, type):
                    target_cls = obj
            except Exception:
                # Fall back to requested cls if anything goes wrong
                target_cls = cls

        # Ensure target_cls is actually a subclass of cls before using it.
        if target_cls is not cls and not issubclass(target_cls, cls):
            raise ValueError(f"Saved experiment class {target_cls} is not a subclass of requested {cls}")

        # Resolve data pipeline path and load it using the BaseDataPipeline logic
        dp_path = exp_dir / exp_config.pop("dp_path")

        # Use BaseDataPipeline.from_disk which will resolve the exact pipeline subclass
        dp = BaseDataPipeline.from_disk(dp_path)

        # Remove temporary keys used for loading
        exp_config.pop("dp_cls", None)
        exp_config.pop("dp_path", None)
        exp_config.pop("exp_cls", None)

        # Ensure exp_dir path is provided to the constructed experiment
        exp_config["exp_dir"] = exp_dir

        # Construct and return the experiment instance (use target_cls)
        exp = target_cls(dp=dp, **exp_config)
        return exp
