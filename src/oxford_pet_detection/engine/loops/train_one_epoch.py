from typing import Any
from collections import defaultdict
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None = None,
    amp: bool = True,
) -> dict[str, float]:

    model.train()

    use_autocast = bool(amp) and (device.type == "cuda")
    use_scaler = use_autocast and (scaler is not None)

    sum_loss = 0.0
    sum_count = 0
    # key 없으면 0.0
    loss_sums = defaultdict(float)

    for images, targets in train_loader:
        # images: list[Tensor], targets: list[dict]
        x = [img.to(device, non_blocking=True) for img in images]

        y: list[dict[str, Any]] = []
        for t in targets:
            d: dict[str, Any] = {}
            # tensor 만
            for k, v in t.items():
                d[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            y.append(d)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_autocast):
            # loss_dict = {"loss_classifier": tensor(0.31), ...}
            loss_dict = model(x, y)
            loss = sum(loss_dict.values())

        if use_scaler:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = len(x)
        sum_loss += loss.item() * bs
        sum_count += bs

        # loss breakdown 누적
        for k, v in loss_dict.items():
            # loss_sums["loss_classifier"] =  0.31 * bs}
            loss_sums[k] += float(v.item()) * bs

    train_loss = sum_loss / max(1, sum_count)
    lr = float(optimizer.param_groups[0]["lr"])

    train_metrics = {
        "train_loss": float(train_loss),
        "lr": lr
    }
    for k, s in loss_sums.items():
        # "loss/loss_classifier": loss_sums["loss_classifier"] / sum_count
        train_metrics[f"loss/{k}"] = float(s / max(1, sum_count))

    return train_metrics