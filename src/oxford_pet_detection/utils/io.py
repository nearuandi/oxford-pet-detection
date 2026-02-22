import io
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import yaml
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def load_image(path_or_url: str) -> np.ndarray:
    if str(path_or_url).startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return np.array(img, dtype=np.uint8)