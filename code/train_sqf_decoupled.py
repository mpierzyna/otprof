from __future__ import annotations

import logging
import pathlib

import lightning as L
import torch

from otprof.logging import get_logger

from experiments import SqueezeFormerExp, get_exp

torch.set_float32_matmul_precision("high")

DEBUG = False

logger = get_logger()


def train(
    exp: SqueezeFormerExp,
    resume_from: str | None = None,
) -> pathlib.Path:
    # Setup data loaders for training and validation
    data_train = exp.get_torch_dataset(stage="train")
    model = exp.get_lit_model(data_train)

    loader_train = exp.get_torch_loader(stage="train", data=data_train)
    loader_val = exp.get_torch_loader(stage="val")
    loader_test = exp.get_torch_loader(stage="test")

    # Train
    trainer = L.Trainer(
        max_epochs=exp.max_epochs,
        devices=[0] if DEBUG else "auto",
        fast_dev_run=DEBUG,
        log_every_n_steps=5,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=5),
        ],
    )
    log_dir = trainer.log_dir or "lightning_logs/debug"
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if trainer.is_global_zero:
        exp.to_disk(log_dir)

    trainer.fit(model, loader_train, loader_val, ckpt_path=resume_from)

    return log_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    train(get_exp("wrf_native"))
    # train(get_exp("wrf_pl"))
    # train(get_exp("era5_pl_direct"))
