import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from oxford_pet_detection.utils import box_iou_xyxy


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    amp: bool = True,
) -> dict[str, float]:

    model.eval()

    use_autocast = bool(amp) and (device.type == "cuda")

    # IoU : Intersection over Union
    # 예측한 박스가 정답 박스랑 얼마나 겹치는지
    # (겹친 영역의 넓이) / (전체 합친 영역의 넓이)
    sum_iou = 0.0
    sum_count = 0
    # IoU가 threshold 이상인 샘플 수
    hit = 0

    for images, targets in val_loader:
        x = [img.to(device, non_blocking=True) for img in images]

        with autocast(device_type=device.type, enabled=use_autocast):
            outputs = model(x)

        for output, target in zip(outputs, targets):
            gt_boxes = target.get("boxes")
            if gt_boxes is None or gt_boxes.numel() == 0:
                continue

            gt_boxes = gt_boxes.to(device, non_blocking=True)
            # ground truth box, 정답 위치 박스
            # [0] 첫 번째 정답 박스만 사용
            gt_box = gt_boxes[0]  # (4,)

            pred_boxes = output.get("boxes", None)
            pred_scores = output.get("scores", None)

            sum_count += 1

            if pred_boxes is None or pred_scores is None or pred_boxes.numel() == 0:
                continue

            pred_boxes = pred_boxes.to(device, non_blocking=True)
            pred_scores = pred_scores.to(device, non_blocking=True)

            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

            if pred_boxes.numel() == 0:
                continue

            # score_threshold 통과한 pred 중에서 GT와 IoU가 가장 큰 값 사용
            ious = box_iou_xyxy(pred_boxes, gt_box.unsqueeze(0)).squeeze(1)  # (N,)
            iou = float(ious.max().item())
            sum_iou += iou

            if iou >= iou_threshold:
                hit += 1

    val_iou_mean = sum_iou / max(1, sum_count)
    val_hit = hit / max(1, sum_count)

    val_metrics = {
        "val_iou_mean": val_iou_mean,
        "val_hit": val_hit,
    }

    return val_metrics