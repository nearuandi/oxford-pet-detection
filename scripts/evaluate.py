import hydra
from omegaconf import DictConfig

from oxford_pet_detection.cli import run_evaluate


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_evaluate(cfg)


if __name__ == "__main__":
    main()