from torchvision.transforms import v2 as T
from omegaconf import DictConfig
import torch

def build_train_transform(cfg: DictConfig):
    dataset = cfg.dataset
    return T.Compose([
        T.Resize((dataset.image_size, dataset.image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
    ])


def build_eval_transform(cfg: DictConfig):
    dataset = cfg.dataset
    return T.Compose([
        T.Resize((dataset.image_size, dataset.image_size)),
        T.ToDtype(torch.float32, scale=True),
    ])