import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------
# binary mask (0/1 이미지) → bounding box (XYXY) 변환
# ---------------------------------------------------------
def masks_to_box_xyxy(mask01: np.ndarray) -> np.ndarray:
    """
    mask01: (H, W) 형태의 0/1 마스크 이미지
    return: [x1, y1, x2, y2] 형태의 바운딩박스 (float32)

    동작:
    - mask에서 값이 1인 좌표들을 찾는다
    - 그 좌표들의 최소/최대값으로 박스를 만든다
    """

    # mask에서 객체(1)인 픽셀 위치 찾기
    ys, xs = np.where(mask01 > 0)

    # 객체가 없는 경우 (빈 마스크)
    # detection pipeline 에러 방지용 fallback box
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    # 좌상단 좌표
    x1 = float(xs.min())
    y1 = float(ys.min())

    # 우하단 좌표
    # +1 하는 이유:
    # 픽셀 인덱스는 inclusive라서 실제 box 크기를 맞추기 위함
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ---------------------------------------------------------
# 두 박스 집합의 IoU 계산 함수
# ---------------------------------------------------------
def box_iou_xyxy(
        boxes1: Tensor,
        boxes2: Tensor
) -> Tensor:
    """
    boxes1: (N,4)  [x1,y1,x2,y2]
    boxes2: (M,4)

    return:
        (N,M) IoU matrix
        → boxes1의 각 박스와 boxes2의 각 박스 간 IoU
    """

    # 각 좌표 분리
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # -------------------------------------------------
    # 교집합(intersection) 영역 계산
    # 브로드캐스팅을 사용해서 (N,M) 형태로 한번에 계산
    # -------------------------------------------------
    xa = torch.max(x11[:, None], x21[None, :])  # 왼쪽 x
    ya = torch.max(y11[:, None], y21[None, :])  # 위쪽 y
    xb = torch.min(x12[:, None], x22[None, :])  # 오른쪽 x
    yb = torch.min(y12[:, None], y22[None, :])  # 아래쪽 y

    # 겹치는 width * height
    # 음수가 나오면 겹치지 않는 것이므로 0으로 clamp
    inter = (xb - xa).clamp(min=0) * (yb - ya).clamp(min=0)

    # -------------------------------------------------
    # 각 박스의 면적(area)
    # -------------------------------------------------
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    # -------------------------------------------------
    # 합집합(union)
    # union = area1 + area2 - intersection
    # -------------------------------------------------
    union = area1[:, None] + area2[None, :] - inter

    # -------------------------------------------------
    # IoU 계산
    # 0 나눗셈 방지를 위해 최소값 clamp
    # 마지막 clamp(0,1)은 float 오차 보정
    # -------------------------------------------------
    return (inter / union.clamp(min=1e-6)).clamp(0, 1)