import argparse
from pathlib import Path

from omegaconf import OmegaConf

from oxford_pet_detection.cli import run_infer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="infer.py",
        description="Oxford Pet Detection - simple inference CLI",
    )
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to Hydra config.yaml")
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt (or any checkpoint payload)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--img", type=str, help="Local image path")
    g.add_argument("--url", type=str, help="Image URL")

    p.add_argument("--score-thr", type=float, default=None, help="Override score threshold")
    p.add_argument("--max-det", type=int, default=None, help="Override max detections")

    p.add_argument("--save", type=str, default=None, help="Save visualization to path")
    p.add_argument("--print-json", action="store_true", help="Print raw prediction dict")
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = OmegaConf.load(args.config)

    if args.score_thr is not None:
        cfg.model.score_thresh = float(args.score_thr)
    if args.max_det is not None:
        cfg.model.max_detections = int(args.max_det)

    src = args.img if args.img is not None else args.url
    run_infer(cfg=cfg, weights_path=Path(args.weights), image_path_or_url=src, save_path=args.save, print_json=args.print_json)


if __name__ == "__main__":
    main()