import io
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf
import requests
import torch
import yaml
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_config(path: Path, cfg) -> None:
    path = Path(path)
    path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

def save_checkpoint(path: Path, ckpt: dict) -> None:
    torch.save(ckpt, path)

def save_history(
        out_dir: str | Path,
        history: dict[str, list],
        train_time: float,
        best_score: float,
        best_metric: str
) -> None:
    out_dir = Path(out_dir)
    history_data = {
        "history": history,
        "train_time": train_time,
        "best_score": best_score,
        "best_metric": best_metric
    }
    torch.save(history_data, out_dir / "history.pt")

def load_image(path_or_url: str) -> Image.Image:
    if str(path_or_url).startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img