import random
from typing import Any
from dataclasses import dataclass
from omegaconf import DictConfig
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor

Target = dict[str, Any]


def _hflip_boxes_xyxy(boxes: Tensor, width: int) -> Tensor:
    if boxes.numel() == 0:
        return boxes
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    boxes[:, 0] = width - x2
    boxes[:, 2] = width - x1
    return boxes


def _resize_boxes_xyxy(boxes: Tensor, orig_w: int, orig_h: int, new_w: int, new_h: int) -> Tensor:
    if boxes.numel() == 0:
        return boxes
    sx = new_w / float(orig_w)
    sy = new_h / float(orig_h)
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return boxes


@dataclass
class DetectionTrainTransform:
    hflip_p: float = 0.5
    jitter: bool = True
    resize_shorter: int = 640  # ✅ 추가 (512~640 추천)

    def __post_init__(self) -> None:
        self.to_tensor = T.ToTensor()
        self.jitter_tf = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)

    def __call__(self, image: Image.Image, target: Target | None = None) -> tuple[torch.Tensor, Target]:
        # 원본 크기
        orig_w, orig_h = image.size

        # ✅ Resize: shorter side를 resize_shorter로 맞춤(비율 유지)
        image = F.resize(image, size=self.resize_shorter)

        new_w, new_h = image.size

        x = self.to_tensor(image)
        if self.jitter:
            x = self.jitter_tf(x)

        # target이 None일 가능성도 방어
        target = dict(target) if target is not None else {}
        if "boxes" in target:
            target["boxes"] = _resize_boxes_xyxy(target["boxes"], orig_w, orig_h, new_w, new_h)

        if random.random() < self.hflip_p:
            _, _, w = x.shape
            x = F.hflip(x)
            if "boxes" in target:
                target["boxes"] = _hflip_boxes_xyxy(target["boxes"], width=w)

        return x, target


@dataclass
class DetectionEvalTransform:
    resize_shorter: int = 640

    def __post_init__(self) -> None:
        self.to_tensor = T.ToTensor()

    def __call__(self, image: Image.Image, target: Target | None = None) -> tuple[torch.Tensor, Target]:
        orig_w, orig_h = image.size
        image = F.resize(image, size=self.resize_shorter)
        new_w, new_h = image.size

        x = self.to_tensor(image)
        target = dict(target) if target is not None else {}
        if "boxes" in target:
            target["boxes"] = _resize_boxes_xyxy(target["boxes"], orig_w, orig_h, new_w, new_h)

        return x, target


def build_train_transform(cfg: DictConfig):
    hflip_p = cfg.train.augment.hflip_p
    jitter = cfg.train.augment.jitter
    resize_shorter = int(getattr(cfg.train.augment, "resize_shorter", 640))
    return DetectionTrainTransform(hflip_p=hflip_p, jitter=jitter, resize_shorter=resize_shorter)


def build_eval_transform(cfg: DictConfig):
    resize_shorter = int(getattr(cfg.train.augment, "resize_shorter", 640))
    return DetectionEvalTransform(resize_shorter=resize_shorter)