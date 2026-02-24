from pathlib import Path
import time

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .loops.train_one_epoch import train_one_epoch
from .loops.evaluate_one_epoch import evaluate_one_epoch
from oxford_pet_detection.utils import ensure_dir, save_checkpoint, save_config, save_history


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)

        self.train = cfg.train

        self.optimizer = AdamW(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=self.train.optimizer.lr,
            weight_decay=self.train.optimizer.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=self.train.scheduler.mode,
            factor=self.train.scheduler.factor,
            patience=self.train.scheduler.patience,
            min_lr=self.train.scheduler.min_lr,
        )

        self.amp_enabled = bool(self.train.amp) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp_enabled)

        self.keep_last = bool(self.train.save.keep_last)

        self.best_metric = "val_iou_mean"
        self.best_score = float("-inf")

    def _make_best_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_score": self.best_score,
            "best_metric": self.best_metric,
        }

    def _make_last_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "best_score": self.best_score,
            "best_metric": self.best_metric,
        }

    def fit(self, out_dir: Path, train_loader: DataLoader, val_loader: DataLoader) -> None:
        out_dir = Path(out_dir)
        ensure_dir(out_dir)
        save_config(out_dir / "config.yaml", self.cfg)

        history = {
            "train_loss": [],
            "val_iou_mean": [],
            "val_hit": [],
        }

        print(f"{self.cfg.exp.name} 훈련 시작")

        start_time = time.time()
        num_epochs = int(self.train.num_epochs)

        for epoch in range(1, num_epochs + 1):
            print(f"[Epoch {epoch:02d}/{num_epochs:02d}] {self.cfg.exp.name}")

            train_metrics = train_one_epoch(
                model=self.model,
                train_loader=train_loader,
                optimizer=self.optimizer,
                device=self.device,
                scaler=self.scaler,
                amp=self.amp_enabled
            )

            val_metrics = evaluate_one_epoch(
                model=self.model,
                val_loader=val_loader,
                device=self.device,
                score_threshold=self.train.metric.score_threshold,
                iou_threshold=self.train.metric.iou_threshold,
                amp=self.amp_enabled
            )

            self.scheduler.step(val_metrics[self.best_metric])

            print(
                f"Train loss={train_metrics['train_loss']:.4f} | "
                f"Val IoU={val_metrics['val_iou_mean']:.4f} | "
                f"Val Hit@{self.train.metric.iou_threshold:.2f}={val_metrics['val_hit']:.4f}"
            )

            history["train_loss"].append(train_metrics["train_loss"])
            history["val_iou_mean"].append(val_metrics["val_iou_mean"])
            history["val_hit"].append(val_metrics["val_hit"])

            score = val_metrics[self.best_metric]

            if score > self.best_score:
                self.best_score = score
                ckpt = self._make_best_payload(epoch=epoch)
                save_checkpoint(out_dir / "best.pt", ckpt)
                print(f"  Best Updated: {self.best_metric}={self.best_score:.4f}")

            if self.keep_last:
                ckpt = self._make_last_payload(epoch=epoch)
                save_checkpoint(out_dir / "last.pt", ckpt)

        train_time = time.time() - start_time
        save_history(
            out_dir=out_dir,
            history=history,
            train_time=train_time,
            best_score=self.best_score,
            best_metric=self.best_metric,
        )

        print(
            f"{self.cfg.exp.name} 훈련 완료, "
            f"train_time: {train_time / 60:.1f}분, "
            f"best_val_iou_mean: {self.best_score:.4f}\n"
        )