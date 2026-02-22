import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)


def _replace_predictor(model: FasterRCNN, num_classes: int) -> FasterRCNN:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_fasterrcnn_resnet50_fpn_v2(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    return _replace_predictor(model, num_classes)


def build_fasterrcnn_mobilenet_v3_large_fpn(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 6,
) -> nn.Module:
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    return _replace_predictor(model, num_classes)