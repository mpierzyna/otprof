from typing import Dict, Tuple
import matplotlib.pyplot as plt

import abc
import lightning as L
import torch

from otprof.losses import pearson_r, edist, ddz_loss
from leap_sqf import SqueezeformerModel
from tensorboardX import SummaryWriter


def log_sample(tb: SummaryWriter, y_true: torch.Tensor, y_pred: torch.Tensor, epoch: int):
    """Log sample predictions of first batch item to TensorBoard."""
    b, l, f = y_true.shape

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


class BaseLitSqueezeFormer(L.LightningModule, abc.ABC):
    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _get_lcn2_metrics(lcn2, lcn2_hat) -> Dict[str, torch.Tensor]:
        """Compute metrics for lcn2 only."""
        mae = torch.abs(lcn2_hat - lcn2).mean()
        rmse = ((lcn2_hat - lcn2) ** 2).mean().sqrt()
        e = edist(lcn2_hat, lcn2, normalize=True)
        r = pearson_r(lcn2, lcn2_hat, dim=1).mean()
        return {"mae_lcn2": mae, "rmse_lcn2": rmse, "edist_lcn2": e, "r_lcn2": r}

    @abc.abstractmethod
    def _shared_step(self, batch, batch_idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Shared step for train/val/test."""

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
        return loss

    def test_step(self, batch, batch_idx):
        loss, log_metrics = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in log_metrics.items():
            self.log(f"test_{k}", v, prog_bar=False, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            # Prep for finetuning based on https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html
            # Only optimize unfrozen parameters
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.lr_tmax)
        return [optimizer], [scheduler]


class LitSqueezeFormer(BaseLitSqueezeFormer):
    def __init__(
        self,
        *,
        input_dim: int,
        levels_in: int,
        output_dim: int,
        levels_out: int,
        embed_dim: int,
        head_dim: int,
        num_blocks: int,
        num_heads: int,
        conv_kernel_size: int,
        use_conf_head: bool = True,
        lr: float = 1e-3,
        lr_tmax: int = 10,
    ):
        super().__init__()
        self.model = SqueezeformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            levels_in=levels_in,
            levels_out=levels_out,
            use_conf_head=use_conf_head,
            embed_dim=embed_dim,
            head_dim=head_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
        )
        print("Compiling model...")
        self.model = torch.compile(self.model)
        print("Done.")
        self.lr = lr
        self.lr_tmax = lr_tmax
        self.save_hyperparameters()

    def _shared_step(self, batch, batch_idx):
        x, y_frc, y = batch  # y shape: (B, H, C)
        y_hat, y_hat_conf = self(x)

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

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        if batch_idx == 0:
            x, _, y = batch
            y_hat, _ = self(x[:2])
            with torch.no_grad():
                log_sample(self.logger.experiment, y, y_hat, epoch=self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            # Prep for finetuning based on https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html
            # Only optimize unfrozen parameters
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]
