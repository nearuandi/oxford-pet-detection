from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from oxford_pet_detection.data import build_datamodule
from oxford_pet_detection.engine import Trainer
from oxford_pet_detection.models import build_model
from oxford_pet_detection.utils import ensure_dir, save_yaml
from oxford_pet_detection.utils import seed_everything


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_train(cfg: DictConfig) -> None:
    seed_everything(int(cfg.train.seed))
    device = resolve_device(str(cfg.train.device))

    out_dir = Path(cfg.paths.out_dir) / str(cfg.exp.name)
    ensure_dir(out_dir)

    save_yaml(out_dir / "config.yaml", OmegaConf.to_container(cfg, resolve=True))

    datamodule = build_datamodule(cfg)

    model = build_model(cfg=cfg).to(device)
    # detection은 내부 loss 계산 -> 별도 loss_fn 불필요
    loss_fn = nn.Identity()

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        cfg=cfg,
        device=device,
    )

    trainer.fit(
        out_dir=out_dir,
        train_loader=datamodule.train_loader,
        val_loader=datamodule.val_loader,
    )