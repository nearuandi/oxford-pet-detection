from pathlib import Path

from omegaconf import DictConfig

from oxford_pet_detection.pipelines import Predictor
from oxford_pet_detection.utils import save_detection_viz


def run_infer(
    cfg: DictConfig,
    weights_path: Path,
    image_path_or_url: str,
    save_path: str | None = None,
    print_json: bool = False,
) -> None:
    predictor = Predictor(cfg=cfg, weights_path=weights_path)
    pred = predictor.predict(image_path_or_url=image_path_or_url)

    if print_json:
        print(pred)

    if save_path is not None:
        save_detection_viz(
            image=pred["image"],
            boxes=pred["boxes"],
            scores=pred["scores"],
            labels=pred["labels"],
            class_names=pred["class_names"],
            save_path=Path(save_path),
        )
        print(f"saved: {save_path}")
    else:
        print(f"num_detections={len(pred['boxes'])}")
        if len(pred["boxes"]) > 0:
            print(f"top1 score={float(pred['scores'][0]):.3f} box={pred['boxes'][0].tolist()}")