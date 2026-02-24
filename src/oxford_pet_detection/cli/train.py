from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from oxford_pet_detection.data import build_datamodule
from oxford_pet_detection.engine import Trainer
from oxford_pet_detection.models import build_model
from oxford_pet_detection.utils import ensure_dir, save_config
from oxford_pet_detection.utils import seed_everything


def run_train(cfg: DictConfig) -> None:
    seed_everything(int(cfg.train.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.paths.out_dir) / cfg.exp.name
    ensure_dir(out_dir)

    save_config(out_dir / "config.yaml", cfg)

    datamodule = build_datamodule(cfg)

    model = build_model(cfg=cfg).to(device)

    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
    )

    trainer.fit(
        out_dir=out_dir,
        train_loader=datamodule.train_loader,
        val_loader=datamodule.val_loader,
    )