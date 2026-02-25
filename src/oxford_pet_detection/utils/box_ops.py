import numpy as np
from torch import Tensor
from torchvision.ops import box_iou


def masks_to_box_xyxy(mask01: np.ndarray) -> np.ndarray:
    # mask[y,x]
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return np.array([0., 0., 1., 1.], dtype=np.float32)
    return np.array([
        xs.min(),
        ys.min(),
        # 끝칸까지 포함되게 만들려고 +1
        xs.max() + 1,
        ys.max() + 1
    ], dtype=np.float32)


def box_iou_xyxy(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
    return box_iou(
        boxes1.float(),
        boxes2.to(boxes1.device).float()).clamp(0, 1)