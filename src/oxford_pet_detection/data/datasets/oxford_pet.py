from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision import tv_tensors

from oxford_pet_detection.utils import masks_to_box_xyxy


@dataclass(frozen=True, slots=True)
class SampleIndex:
    image_path: Path
    mask_path: Path
    species_id: int  # 1=cat, 2=dog


LabelMode = Literal["pet", "species"]


class OxfordPetDetectionDataset(Dataset):
    def __init__(
        self,
        samples: list[SampleIndex],
        label_mode: LabelMode = "pet",
        transform=None,
    ) -> None:
        self.samples = samples
        self.label_mode = label_mode
        self.transform = transform

        if self.label_mode == "pet":
            self.num_classes = 2
            self.class_names = ["__background__", "pet"]
        else:
            self.num_classes = 3
            self.class_names = ["__background__", "cat", "dog"]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> PILImage.Image:
        return PILImage.open(path).convert("RGB")

    def _load_mask(self, path: Path) -> np.ndarray:
        return np.array(PILImage.open(path), dtype=np.uint8)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        img_pil = self._load_image(s.image_path)
        mask = self._load_mask(s.mask_path)

        obj = (mask != 2).astype(np.uint8)
        box = masks_to_box_xyxy(obj)

        # 0, 1
        h, w = mask.shape[:2]
        boxes = tv_tensors.BoundingBoxes(
            torch.from_numpy(box).to(torch.float32).unsqueeze(0),
            format="XYXY",
            canvas_size=(h, w),
        )

        if self.label_mode == "pet":
            label = 1
        else:
            label = 1 if s.species_id == 1 else 2

        target = {
            "boxes": boxes,
            "labels": torch.tensor([label], dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "iscrowd": torch.zeros((1,), dtype=torch.int64),
            "area": torch.tensor([(box[2] - box[0]) * (box[3] - box[1])], dtype=torch.float32),
            "path": str(s.image_path),
        }

        if self.transform is not None:
            image, target = self.transform(img_pil, target)

            # resize / flip 후 area 다시 계산
            b = target["boxes"][0]
            target["area"] = torch.tensor(
                [(b[2] - b[0]) * (b[3] - b[1])],
                dtype=torch.float32,
            )
        else:
            image = torch.from_numpy(np.array(img_pil, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

        target["class_names"] = self.class_names
        return image, target