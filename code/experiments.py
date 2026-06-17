from __future__ import annotations

import lightning as L
import torch

import otprof.pipelines as pipelines
from otprof.base import BaseExperiment
from otprof.datasets import ProfileDataset
from otprof.logging import get_logger
from otprof.types import TStage
from lit import LitSqueezeFormer

logger = get_logger()


class SqueezeFormerExp(BaseExperiment):
    embed_dim: int  # capacity of squeezeformer and attention layers
    head_dim: int  # dimensionality of final prediction head
    num_blocks: int  # depth, ie. number of squeezeformer blocks
    num_heads: int  # number of attention heads
    conv_kernel_size: int  # kernel size of convolutional layers in squeezeformer
    use_conf_head: bool  # use_conf_head

    debug: bool = False

    def get_torch_dataset(self, stage: TStage) -> ProfileDataset:
        ds_X, ds_y = self.dp.split(transform=True, stage=stage)
        return ProfileDataset(
            ds_X=ds_X,
            ds_y=ds_y,
            vars_X=self.features,
            vars_y=self.targets,
            vars_y_frc=self.forcings,
        )

    def get_torch_loader(self, stage: TStage, data: ProfileDataset | None = None) -> torch.utils.data.DataLoader:
        data = data or self.get_torch_dataset(stage=stage)
        shuffle = stage == "train"

        return torch.utils.data.DataLoader(
            data,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=1 if self.debug else 16,
            persistent_workers=True,
        )

    def get_lit_model(self, data_train: ProfileDataset) -> L.LightningModule:
        levels_X = data_train.n_levels_X
        levels_y = data_train.n_levels_y
        logger.info(f"Input levels: {levels_X}, Output levels: {levels_y}")

        return LitSqueezeFormer(
            input_dim=data_train.n_vars_X,
            levels_in=levels_X,
            output_dim=data_train.n_vars_y,
            levels_out=levels_y,
            use_conf_head=self.use_conf_head,
            embed_dim=self.embed_dim,
            head_dim=self.head_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            conv_kernel_size=self.conv_kernel_size,
            lr=1e-3,
            lr_tmax=self.max_epochs,
        )


def get_exp(name: str) -> SqueezeFormerExp:
    """Get experiment by name. Keep experiments in function to avoid pipelines loading on import."""
    # Base experiment: WRF native to WRF
    wrf_native = SqueezeFormerExp(
        name="sqf_wrfnative_wrf",
        dp=pipelines.wrf_wrf.p_wrf_orig,  # NATIVE WRF to WRF
        # Features / Target
        features=[
            "z_agl",
            "p",
            "u",
            "v",
            # "w",
            "th",
            # Gradients
            "S",
            "dth_dz",
            # Surface
            "ust",
            "hfx",
            "lh",
            "blh",
            "u10",
            "v10",
            "tk2",
            "slp",
            "lsm",
            # Time features
            "hr_sin",
            "hr_cos",
            "doy_sin",
            "doy_cos",
            # surface estimation of CT2
            "lct2_w71f",
        ],
        forcings=["z_agl"],
        targets=[
            "lcn2",
            "QKE",  # weird distribution
            "TSQ",
            "EL_PBL",
        ],
        # Model settings
        use_conf_head=True,
        conv_kernel_size=3,
        embed_dim=48,
        num_blocks=4,
        num_heads=2,
        head_dim=512,
        # Training settings
        batch_size=8 * 1024,
        max_epochs=300,
    )

    ## Study with progressively more realistic data pipelines
    if name == "wrf_native":
        return wrf_native
    elif name == "wrf_pl":
        # Training with WRF PL levels
        return wrf_native.model_copy(
            update={
                "name": "sqf_wrfpl_wrf",
                "dp": pipelines.wrf_wrf.p_wrfpl_wrf.setup(),
                "batch_size": 18 * 1024,
                "max_epochs": 600,  # very long training needed
            }
        )
    elif name == "era5_pl_direct":
        # direct era5 training
        return wrf_native.model_copy(
            update={
                "name": "sqf_era5pl_wrf_direct",
                "dp": pipelines.era5_wrf.p_era5pl_wrf.setup(),
                "batch_size": 18 * 1024,
                "max_epochs": 600,
            }
        )
    else:
        raise ValueError(f"Unknown experiment name: {name}")
