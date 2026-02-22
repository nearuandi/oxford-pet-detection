from collections.abc import Iterable

import torch
import torch.nn as nn

from oxford_pet_detection.utils import box_iou_xyxy


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,  # unused (kept)
    data_loader: Iterable,
    device: torch.device,
    score_thr: float = 0.5,
    iou_thr: float = 0.5,
) -> dict:
    model.eval()

    ious: list[float] = []
    correct = 0
    total = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        # model(images) -> detections
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_box = tgt["boxes"][0].to(device)  # (4,)
            pred_boxes = out["boxes"]
            pred_scores = out["scores"]

            keep = pred_scores >= score_thr
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

            total += 1
            if pred_boxes.numel() == 0:
                ious.append(0.0)
                continue

            # top-1
            top_idx = int(torch.argmax(pred_scores).item())
            pb = pred_boxes[top_idx]

            iou = float(box_iou_xyxy(pb.unsqueeze(0), gt_box.unsqueeze(0)).item())
            ious.append(iou)
            if iou >= iou_thr:
                correct += 1

    iou_mean = float(sum(ious) / max(len(ious), 1))
    precision = float(correct / max(total, 1))
    return {
        "val_iou_mean": iou_mean,
        "val_precision": precision,
    }