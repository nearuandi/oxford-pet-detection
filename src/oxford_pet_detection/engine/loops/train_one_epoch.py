from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    amp: bool,
    log_interval: int = 20,
) -> dict:

    model.train()

    total_loss = 0.0
    n = 0

    for step, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items() if k != "class_names"} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            with autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n += 1

        if log_interval > 0 and step % log_interval == 0:
            print(f"  step={step:04d} loss={float(loss.item()):.4f}")

    return {"train_loss": total_loss / max(n, 1)}