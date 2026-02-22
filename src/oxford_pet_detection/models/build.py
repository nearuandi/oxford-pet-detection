from omegaconf import DictConfig

from oxford_pet_detection.models.fasterrcnn import (
    build_fasterrcnn_mobilenet_v3_large_fpn,
    build_fasterrcnn_resnet50_fpn_v2,
)


def build_model(cfg: DictConfig):
    name = str(cfg.model.name)
    num_classes = int(cfg.dataset.num_classes)

    pretrained = bool(cfg.model.pretrained)
    trainable_backbone_layers = int(cfg.model.trainable_backbone_layers)

    if name == "fasterrcnn_resnet50_fpn_v2":
        return build_fasterrcnn_resnet50_fpn_v2(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    if name == "fasterrcnn_mobilenet_v3_large_fpn":
        return build_fasterrcnn_mobilenet_v3_large_fpn(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    raise ValueError(f"Unknown model.name: {name}")