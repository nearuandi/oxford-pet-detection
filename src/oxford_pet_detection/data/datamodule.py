from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .collate import detection_collate
from .datasets.oxford_pet import OxfordPetDetectionDataset, SampleIndex
from .transforms import build_eval_transform, build_train_transform
from oxford_pet_detection.utils import make_rng


@dataclass(frozen=True, slots=True)
class DataModule:
    train_loader: DataLoader
    val_loader: DataLoader


def read_official_trainval(
        root: Path
) -> list[tuple[str, int]]:
    p = root / "annotations" / "trainval.txt"

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
    # stem cat.jpg -> cat
    return sorted(p.stem for p in (root / "images").glob("*.jpg"))


def build_samples(cfg: DictConfig) -> list[SampleIndex]:
    dataset = cfg.dataset
    root = Path(dataset.root)
    # images
    images_dir = root / dataset.images_dir
    # annotations/trimaps
    masks_dir = root / dataset.masks_dir

    pairs = read_official_trainval(root)
    if not pairs:
        raise FileNotFoundError("annotations/trainval.txt 에러")

    samples: list[SampleIndex] = []
    for name, species_id in pairs:
        image_path = images_dir / f"{name}.jpg"
        mask_path = masks_dir / f"{name}.png"

        if image_path.exists() and mask_path.exists():
            samples.append(
                SampleIndex(
                    image_path=image_path,
                    mask_path=mask_path,
                    species_id=species_id
                )
            )
    return samples


def split_samples(
        cfg: DictConfig,
        samples: list[SampleIndex]
) -> tuple[list[SampleIndex], list[SampleIndex]]:
    rng = make_rng(cfg.dataset.seed)

    idx = np.arange(len(samples))
    rng.shuffle(idx)

    n_train = int(len(idx) * float(cfg.dataset.train_ratio))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def build_datamodule(cfg: DictConfig) -> DataModule:
    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    samples = build_samples(cfg)
    train_samples, val_samples = split_samples(cfg, samples)

    train_ds = OxfordPetDetectionDataset(
        samples=train_samples,
        label_mode=cfg.dataset.label_mode,
        transform=train_transform,
    )
    val_ds = OxfordPetDetectionDataset(
        samples=val_samples,
        label_mode=cfg.dataset.label_mode,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        collate_fn=detection_collate,
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        collate_fn=detection_collate,
    )

    return DataModule(train_loader=train_loader, val_loader=val_loader)