from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from oxford_pet_detection.engine.loops.evaluate_one_epoch import evaluate_one_epoch
from oxford_pet_detection.engine.loops.train_one_epoch import train_one_epoch
from oxford_pet_detection.utils import ensure_dir, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device

        self.optimizer = AdamW(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=float(cfg.train.optimizer.lr),
            weight_decay=float(cfg.train.optimizer.weight_decay),
        )

        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=str(cfg.train.scheduler.mode),
            factor=float(cfg.train.scheduler.factor),
            patience=int(cfg.train.scheduler.patience),
            min_lr=float(cfg.train.scheduler.min_lr),
        )

        self.amp = bool(cfg.train.amp)
        self.scaler = GradScaler(enabled=self.amp)

        self.monitor = str(cfg.train.save.monitor)
        self.monitor_mode = str(cfg.train.save.mode)

        self.best_score = -1e9 if self.monitor_mode == "max" else 1e9

    def _is_better(self, score: float) -> bool:
        return score > self.best_score if self.monitor_mode == "max" else score < self.best_score

    def fit(self, out_dir: Path, train_loader: DataLoader, val_loader: DataLoader) -> None:
        ensure_dir(out_dir)

        for epoch in range(1, int(self.cfg.train.num_epochs) + 1):
            print(f"[Epoch {epoch:03d}/{int(self.cfg.train.num_epochs):03d}] {self.cfg.exp.name}")

            train_metrics = train_one_epoch(
                model=self.model,
                loss_fn=self.loss_fn,
                data_loader=train_loader,
                optimizer=self.optimizer,
                device=self.device,
                scaler=self.scaler,
                amp=self.amp,
                log_interval=int(self.cfg.train.log_interval),
            )

            val_metrics = evaluate_one_epoch(
                model=self.model,
                loss_fn=self.loss_fn,
                data_loader=val_loader,
                device=self.device,
                score_thr=float(self.cfg.train.metric.score_thr),
                iou_thr=float(self.cfg.train.metric.iou_thr),
            )

            score = float(val_metrics[self.monitor])
            self.scheduler.step(score)

            msg = (
                f"  train_loss={train_metrics['train_loss']:.4f} | "
                f"val_iou_mean={val_metrics['val_iou_mean']:.4f} | "
                f"val_precision={val_metrics['val_precision']:.4f}"
            )
            print(msg)

            payload = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_score": self.best_score,
                "monitor": self.monitor,
                "metrics": {**train_metrics, **val_metrics},
            }

            # last
            if bool(self.cfg.train.save.keep_last):
                save_checkpoint(out_dir / "last.pt", payload)

            # best
            if self._is_better(score):
                self.best_score = score
                payload["best_score"] = self.best_score
                save_checkpoint(out_dir / "best.pt", payload)
                print(f"  best updated: {self.monitor}={self.best_score:.4f}")