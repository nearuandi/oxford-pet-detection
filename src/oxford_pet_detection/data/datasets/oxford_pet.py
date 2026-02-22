from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image as PILImage
from torch.utils.data import Dataset

from oxford_pet_detection.utils.box_ops import masks_to_box_xyxy
from oxford_pet_detection.utils.seed import make_rng





@dataclass(frozen=True, slots=True)
class SampleIndex:
    image_path: Path
    mask_path: Path
    species_id: int  # 1=cat, 2=dog (official trainval.txt 기준)


def read_official_trainval(root: Path) -> list[tuple[str, int]]:
    p = root / "annotations" / "trainval.txt"
    if not p.exists():
        return []
    rows: list[tuple[str, int]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        image_name = parts[0]
        species_id = int(parts[2])
        rows.append((image_name, species_id))
    return rows


def list_all_images(root: Path) -> list[str]:
    images_dir = root / "images"
    return sorted([p.stem for p in images_dir.glob("*.jpg")])

Split = Literal["train", "val"]

class OxfordPetDetectionDataset(Dataset):
    """
    Oxford-IIIT Pet trimaps -> single bbox target
    image당 pet 객체 1개 가정
    """

    def __init__(self, cfg: DictConfig, split: Split, transform=None) -> None:
        self.cfg = cfg
        self.split = split
        self.transform = transform

        root = Path(str(cfg.dataset.root))
        images_dir = root / str(cfg.dataset.images_dir)
        masks_dir = root / str(cfg.dataset.masks_dir)

        label_mode = str(cfg.dataset.label_mode)
        if label_mode not in {"pet", "species"}:
            raise ValueError(f"Unsupported label_mode: {label_mode}")

        official = bool(cfg.dataset.use_official_trainval)
        if official:
            pairs = read_official_trainval(root)
            if not pairs:
                names = list_all_images(root)
                pairs = [(n, 1) for n in names]
        else:
            names = list_all_images(root)
            pairs = [(n, 1) for n in names]

        samples: list[SampleIndex] = []
        for name, species_id in pairs:
            img_p = images_dir / f"{name}.jpg"
            msk_p = masks_dir / f"{name}.png"
            if img_p.exists() and msk_p.exists():
                samples.append(SampleIndex(image_path=img_p, mask_path=msk_p, species_id=species_id))

        rng = make_rng(int(cfg.dataset.seed))
        idx = np.arange(len(samples))
        rng.shuffle(idx)

        train_ratio = float(cfg.dataset.train_ratio)
        n_train = int(len(idx) * train_ratio)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]

        self.samples = [samples[i] for i in (train_idx if split == "train" else val_idx)]

        self.label_mode = label_mode
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
        m = PILImage.open(path)
        return np.array(m, dtype=np.uint8)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_pil = self._load_image(s.image_path)
        mask = self._load_mask(s.mask_path)

        obj = (mask != 2).astype(np.uint8)  # background 제외
        box = masks_to_box_xyxy(obj)        # (4,)
        boxes = torch.tensor([box], dtype=torch.float32)

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

        # ✅ 핵심 변경: image + target 함께 transform
        if self.transform is not None:
            image, target = self.transform(img_pil, target)
        else:
            # fallback: 최소 tensor 변환만
            image = torch.from_numpy(np.array(img_pil, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

        target["class_names"] = self.class_names
        return image, target