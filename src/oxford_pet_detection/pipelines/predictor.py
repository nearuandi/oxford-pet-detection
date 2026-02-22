from pathlib import Path

import torch
from omegaconf import DictConfig

from oxford_pet_detection.models import build_model
from oxford_pet_detection.utils import load_image


class Predictor:
    def __init__(self, cfg: DictConfig, weights_path: Path, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(cfg).to(self.device)

        payload = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

        self.score_thr = float(getattr(cfg.model, "score_thresh", 0.5))
        self.max_det = int(getattr(cfg.model, "max_detections", 50))

        # class names
        self.class_names = list(cfg.dataset.class_names)

    @torch.inference_mode()
    def predict(self, image_path_or_url: str) -> dict:
        img_pil = load_image(image_path_or_url)
        x = torch.from_numpy(img_pil).permute(2, 0, 1).float() / 255.0
        x = x.to(self.device)

        out = self.model([x])[0]
        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()
        labels = out["labels"].detach().cpu()

        keep = scores >= self.score_thr
        boxes = boxes[keep][: self.max_det]
        scores = scores[keep][: self.max_det]
        labels = labels[keep][: self.max_det]

        return {
            "image": img_pil,  # np.ndarray (H,W,3) uint8
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "class_names": self.class_names,
        }