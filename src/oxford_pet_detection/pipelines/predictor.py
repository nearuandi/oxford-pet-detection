import torch
from omegaconf import DictConfig
from oxford_pet_detection.data import build_eval_transform
from oxford_pet_detection.models import build_model
from oxford_pet_detection.utils import load_image


class Predictor:
    def __init__(
            self,
            cfg: DictConfig,
            ckpt: dict,
            device: torch.device | None = None
    ) -> None:
        self.cfg = cfg
        self.device = device

        self.model = build_model(cfg)
        self.model.to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        self.model.eval()

        self.score_threshold = cfg.model.score_thresh
        self.max_detections = cfg.model.max_detections

        self.transform = build_eval_transform(cfg)
        self.dataset = cfg.dataset

    @torch.inference_mode()
    def predict(self, image_path_or_url: str) -> dict:
        img = load_image(image_path_or_url)
        x, _ = self.transform(img, None)
        x = x.to(self.device)

        out = self.model([x])[0]

        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()
        labels = out["labels"].detach().cpu()

        keep = scores >= self.score_threshold
        boxes = boxes[keep][: self.max_detections]
        scores = scores[keep][: self.max_detections]
        labels = labels[keep][: self.max_detections]

        # (C,H,W) -> (H,W,C)
        viz = x.detach().cpu().permute(1, 2, 0).numpy() * 255.0
        viz = viz.clip(0, 255).astype("uint8")

        return {
            "image": viz,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "class_names": list(self.dataset.class_names),  # (json 저장도 안전)
        }