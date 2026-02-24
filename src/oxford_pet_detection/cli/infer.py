from pathlib import Path
import json
import torch
from omegaconf import DictConfig

from oxford_pet_detection.pipelines import Predictor
from oxford_pet_detection.utils import save_detection_viz, ensure_dir


def run_infer(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.paths.out_dir) / str(cfg.exp.name)
    best_path = out_dir / "best.pt"

    infer_dir = out_dir / "infer"
    ensure_dir(infer_dir)

    png_path = infer_dir / "image.png"
    json_path = infer_dir / "infer.json"

    ckpt = torch.load(best_path, map_location=device)

    image_path = str(cfg.image_path)
    predictor = Predictor(cfg=cfg, ckpt=ckpt, device=device)
    result = predictor.predict(image_path_or_url=image_path)

    num_detections = len(result["boxes"])
    print(f"num_detections={num_detections}")
    if num_detections > 0:
        top1_score = float(result["scores"][0])
        top1_box = result["boxes"][0].tolist()
        top1_label = int(result["labels"][0])
        top1_name = result["class_names"][top1_label] if top1_label < len(result["class_names"]) else str(top1_label)
        print(f"top1: label={top1_label}({top1_name}) score={top1_score:.3f} box={top1_box}")
    else:
        print("no detections above threshold")

    save_detection_viz(
        image=result["image"],
        boxes=result["boxes"],
        scores=result["scores"],
        labels=result["labels"],
        class_names=result["class_names"],
        save_path=png_path,
    )
    print(f"saved: {png_path}")

    payload = {
        "image_path_or_url": image_path,
        "boxes": result["boxes"].tolist(),
        "scores": result["scores"].tolist(),
        "labels": result["labels"].tolist(),
        "class_names": list(result["class_names"])
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {json_path}")