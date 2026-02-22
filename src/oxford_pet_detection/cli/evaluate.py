from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from oxford_pet_detection.data import build_datamodule
from oxford_pet_detection.engine import evaluate_one_epoch
from oxford_pet_detection.models import build_model
from oxford_pet_detection.utils import seed_everything


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluate(cfg: DictConfig) -> None:
    seed_everything(int(cfg.train.seed))
    device = resolve_device(str(cfg.train.device))

    datamodule = build_datamodule(cfg)
    model = build_model(cfg=cfg).to(device)

    exp_dir = Path(cfg.paths.out_dir) / str(cfg.exp.name)
    best_path = exp_dir / "best.pt"
    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    dummy_loss = nn.Identity()
    metrics = evaluate_one_epoch(
        model=model,
        loss_fn=dummy_loss,
        data_loader=datamodule.val_loader,
        device=device,
        score_thr=float(cfg.train.metric.score_thr),
        iou_thr=float(cfg.train.metric.iou_thr),
    )

    print(f"[EVAL] exp={cfg.exp.name} | val_iou_mean={metrics['val_iou_mean']:.4f} | val_precision@{cfg.train.metric.iou_thr}={metrics['val_precision']:.4f}")