from dataclasses import dataclass

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .collate import detection_collate
from .datasets.oxford_pet import OxfordPetDetectionDataset
from .transforms import build_eval_transform, build_train_transform


@dataclass(frozen=True, slots=True)
class DataModule:
    train_loader: DataLoader
    val_loader: DataLoader


def build_datamodule(cfg: DictConfig) -> DataModule:
    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    train_ds = OxfordPetDetectionDataset(cfg=cfg, split="train", transform=train_transform)
    val_ds = OxfordPetDetectionDataset(cfg=cfg, split="val", transform=eval_transform)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
        persistent_workers=bool(cfg.train.persistent_workers) if int(cfg.train.num_workers) > 0 else False,
        collate_fn=detection_collate,
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
        persistent_workers=bool(cfg.train.persistent_workers) if int(cfg.train.num_workers) > 0 else False,
        collate_fn=detection_collate,
    )

    return DataModule(train_loader=train_loader, val_loader=val_loader)