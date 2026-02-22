import random
from typing import Any
from dataclasses import dataclass
from omegaconf import DictConfig
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image




# horizontal flip 할 때 bounding box 좌표를 같이 뒤집는 함수
def _hflip_boxes_xyxy(
        boxes: torch.Tensor,
        width: int
) -> torch.Tensor:
    """
    boxes: (N,4) in XYXY
    width: image width (W)
    """
    # 텐서에 들어있는 총 값의 개수를 반환
    if boxes.numel() == 0:
        return boxes
    # PyTorch tensor는 reference라서
    # 직접 쓰면 값이 꼬일 수 있음
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    boxes[:, 0] = width - x2
    boxes[:, 2] = width - x1
    return boxes

Target = dict[str, Any]

@dataclass
class DetectionTrainTransform:
    hflip_p: float = 0.5
    jitter: bool = True

    def __post_init__(self) -> None:
        self.to_tensor = T.ToTensor()  # PIL -> float32 tensor [0,1], (C,H,W)
        self.jitter_tf = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)

    def __call__(
            self,
            image: Image.Image,
            target: Target
    ) -> tuple[torch.Tensor, Target]:
        x = self.to_tensor(image)  # (C,H,W), float32
        # (optional) jitter
        if self.jitter:
            x = self.jitter_tf(x)

        # (optional) horizontal flip (image + boxes sync)
        if random.random() < self.hflip_p:
            _, _, w = x.shape
            x = F.hflip(x)
            target = dict(target)  # shallow copy
            target["boxes"] = _hflip_boxes_xyxy(target["boxes"].clone(), width=w)

        return x, target


@dataclass
class DetectionEvalTransform:
    def __post_init__(self) -> None:
        self.to_tensor = T.ToTensor()

    def __call__(
            self,
            image: Image.Image,
            target: Target
    ) -> tuple[torch.Tensor, Target]:
        x = self.to_tensor(image)
        return x, target


def build_train_transform(cfg: DictConfig):
    hflip_p = cfg.train.augment.hflip_p
    jitter = cfg.train.augment.jitter
    return DetectionTrainTransform(hflip_p=hflip_p, jitter=jitter)


def build_eval_transform(cfg: DictConfig):
    return DetectionEvalTransform()