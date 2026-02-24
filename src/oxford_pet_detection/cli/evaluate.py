from pathlib import Path
import torch
from omegaconf import DictConfig

from oxford_pet_detection.data import build_datamodule
from oxford_pet_detection.engine import evaluate_one_epoch
from oxford_pet_detection.models import build_model
from oxford_pet_detection.utils import seed_everything


def run_evaluate(cfg: DictConfig) -> None:
    seed_everything(cfg.train.seed)
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

    datamodule = build_datamodule(cfg)
    model = build_model(cfg=cfg).to(device)

    exp_dir = Path(cfg.paths.out_dir) / str(cfg.exp.name)
    best_path = exp_dir / "best.pt"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)

    best_epoch = ckpt.get("epoch")
    best_metric = ckpt.get("best_metric")
    best_score = ckpt.get("best_score")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train = cfg.train

    val_metrics = evaluate_one_epoch(
        model=model,
        val_loader=datamodule.val_loader,
        device=device,
        score_threshold=train.metric.score_threshold,
        iou_threshold=train.metric.iou_threshold,
        amp=train.amp
    )

    print(f"cfg.model.name: {cfg.model.name}")
    print(f"cfg.exp.name= {cfg.exp.name}")
    print(f"BEST epoch={best_epoch:02d} | best_metric={best_metric} | best_score={best_score:.2f}")
    print(f"Val IoU={val_metrics['val_iou_mean']:.4f} | ")
    print(f"Val Hit@{train.metric.iou_threshold:.2f}={val_metrics['val_hit']:.4f}")