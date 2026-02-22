import numpy as np
import torch


def masks_to_box_xyxy(mask01: np.ndarray) -> np.ndarray:
    """
    mask01: (H,W) uint8 {0,1}
    returns: [x1,y1,x2,y2] in pixel coords (float)
    """
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        # fallback: empty -> 0 box
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (N,4), boxes2: (M,4)
    returns: (N,M)
    """
    # intersection
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    xa = torch.max(x11[:, None], x21[None, :])
    ya = torch.max(y11[:, None], y21[None, :])
    xb = torch.min(x12[:, None], x22[None, :])
    yb = torch.min(y12[:, None], y22[None, :])

    inter = (xb - xa).clamp(min=0) * (yb - ya).clamp(min=0)

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)