"""SQF-H training: the Lightning module around `SqueezeformerHModel` (loss, metrics, optimizer, sample plots).

`get_lit_model(config, data_train)` builds it from an experiment config and the training `ProfileDataset`. Checkpoints
hold plain hparams only, so `otprof.models.bundle.load_network_from_checkpoint` reads them without this module.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import lightning as L
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter

from otprof.losses import ddz_loss, edist
from otprof.models.sqf_h.model import FORCINGS, SqueezeformerHModel

logger = logging.getLogger(__name__)


def log_sample(tb: SummaryWriter, y_true: torch.Tensor, y_pred: torch.Tensor, epoch: int):
    """Log sample predictions of first batch item to TensorBoard."""
    f = y_true.shape[-1]

    # Plot
    fig, axarr = plt.subplots(ncols=f, nrows=1, figsize=(2 * f, 2), constrained_layout=True)
    if f == 1:
        axarr = [axarr]
    for i, ax in enumerate(axarr):
        ax.plot(y_true[0, :, i].detach().cpu().numpy(), label="true")
        ax.plot(y_pred[0, :, i].detach().cpu().numpy(), label="pred")
        ax.set_title(f"Feature {i}")
        ax.set_xlabel("Level")
        ax.set_ylabel("Value")
        ax.legend()

    tb.add_figure("true_pred_sample", fig, global_step=epoch)
    plt.close(fig)


class LitSqueezeFormerH(L.LightningModule):
    """Height-aware SqueezeFormer (SQF-H). Forcings must be `[z_agl, lz]` (`model.FORCINGS`): `ddz_loss` uses z_agl,
    the model queries at lz. Training predicts a random sorted subset of `n_query_train` output
    levels per batch so the head learns a function of height instead of the level index;
    val/test/predict use all levels.

    `vars_X`, `forcings`, `targets` (variable order of the tensors) are only recorded in the checkpoint, for exporting
    the model; `compile` wraps the network in `torch.compile` (weights then under `model._orig_mod.`)."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        head_dim: int,
        num_blocks: int,
        num_heads: int,
        conv_kernel_size: int,
        idx_valid: int,
        idx_lz: int,
        use_conf_head: bool = True,
        n_query_train: int = 32,
        lr: float = 1e-3,
        lr_tmax: int = 10,
        vars_X: list[str] | None = None,
        forcings: list[str] | None = None,
        targets: list[str] | None = None,
        compile: bool = True,
    ):
        super().__init__()
        self.model = SqueezeformerHModel(
            input_dim=input_dim,
            output_dim=output_dim,
            use_conf_head=use_conf_head,
            embed_dim=embed_dim,
            head_dim=head_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            idx_valid=idx_valid,
            idx_lz=idx_lz,
        )
        if compile:
            logger.info("Compiling the network (torch.compile)")
            self.model = torch.compile(self.model)
        self.n_query_train = n_query_train
        self.lr = lr
        self.lr_tmax = lr_tmax
        self.save_hyperparameters()

    def forward(self, x, frc):
        return self.model(x, frc)

    def _shared_step(self, batch, batch_idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x, y_frc, y = batch  # y shape: (B, H, C), y_frc: (B, H, [z_agl, lz])
        if self.training:
            # Random sorted query subset, shared across the batch (static shapes for compile)
            q = torch.randperm(y.shape[1], device=y.device)[: self.n_query_train].sort().values
            y, y_frc = y[:, q], y_frc[:, q]
        y_hat, y_hat_conf = self(x, y_frc)

        # RMSE
        sq_err_y = (y_hat - y) ** 2
        rmse = torch.sqrt(sq_err_y.mean())

        # Confidence error
        abs_err_conf = torch.abs(y_hat_conf - sq_err_y)
        rmse_conf = torch.sqrt(abs_err_conf.mean())

        loss = rmse + rmse_conf

        edist_y = edist(y_hat, y, normalize=True)
        ddz_mse = ddz_loss(y_hat, y, y_frc[:, :, 0]) / 1e3
        return loss, {
            "mae_loss": torch.abs(y_hat - y).mean(),
            "rmse_loss": rmse,
            "edist": edist_y,
            "ddz_mse": ddz_mse,
        }

    def training_step(self, batch, batch_idx):
        loss, log_metrics = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in log_metrics.items():
            self.log(f"train_{k}", v, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_metrics = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in log_metrics.items():
            self.log(f"val_{k}", v, prog_bar=False, sync_dist=True)
        if batch_idx == 0:
            x, y_frc, y = batch
            y_hat, _ = self(x[:2], y_frc[:2])
            with torch.no_grad():
                log_sample(self.logger.experiment, y, y_hat, epoch=self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, log_metrics = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in log_metrics.items():
            self.log(f"test_{k}", v, prog_bar=False, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y_frc, _ = batch
        return self(x, y_frc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),  # only unfrozen parameters (fine-tuning)
            lr=self.lr,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]


def get_lit_model(config, data_train, compile: bool = True) -> LitSqueezeFormerH:
    """SQF-H of an experiment config (`otprof.experiment.ExperimentConfig`) for the training `ProfileDataset`
    (input / output sizes; `valid` and `lz` looked up by name, the dataset puts surface variables first)."""
    if data_train.vars_y_frc != FORCINGS:
        raise ValueError(f"SQF-H needs the forcings {FORCINGS}, got {data_train.vars_y_frc}")
    m = config.model
    return LitSqueezeFormerH(
        input_dim=data_train.n_vars_X,
        output_dim=data_train.n_vars_y,
        use_conf_head=m.use_conf_head,
        embed_dim=m.embed_dim,
        head_dim=m.head_dim,
        num_blocks=m.num_blocks,
        num_heads=m.num_heads,
        conv_kernel_size=m.conv_kernel_size,
        idx_valid=data_train.vars_X.index("valid"),
        idx_lz=data_train.vars_X.index("lz"),
        n_query_train=m.n_query_train,
        lr=config.training.lr,
        lr_tmax=config.training.max_epochs,
        vars_X=data_train.vars_X,
        forcings=data_train.vars_y_frc,
        targets=data_train.vars_y,
        compile=compile,
    )
