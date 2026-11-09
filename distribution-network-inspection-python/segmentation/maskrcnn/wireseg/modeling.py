from pathlib import Path

import torch
from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from .common import PROJECT_ROOT


def build_model(config, *, initialize_pretrained=False):
    specification, evaluation = config["model"], config["evaluation"]
    # Explicit FrozenBatchNorm keeps the COCO-pretrained and restored topology identical.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=FrozenBatchNorm2d,
        trainable_layers=specification["trainable_backbone_layers"],
    )
    model = MaskRCNN(
        backbone,
        num_classes=91,
        min_size=specification["input_size"],
        max_size=specification["input_size"],
        box_score_thresh=evaluation["score_threshold"],
        box_nms_thresh=specification["nms_threshold"],
        box_detections_per_img=evaluation["max_detections"],
    )
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = 0.0  # Match torchvision's COCO_V1 checkpoint recipe.
    if initialize_pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        directory = PROJECT_ROOT / "weights/pretrained"
        directory.mkdir(parents=True, exist_ok=True)
        state = torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=str(directory),
            map_location="cpu",
            check_hash=True,
            progress=True,
        )
        model.load_state_dict(state)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(channels, 256, 2)
    return model
