"""
Load the Faster R-CNN dental anomaly detector for CPU inference.

Supports dental_model_v3.pth with automatic fallback to dental_model_v2.pth.
"""
import logging
import os

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import MODEL_PATH, MODEL_URL, MODEL_V2_PATH, MODEL_V3_PATH, NUM_CLASSES, resolve_model_path
from download_model import ensure_model_file, get_model_file_info

logger = logging.getLogger(__name__)


def build_model(num_classes: int = NUM_CLASSES):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_model_version_label(path: str) -> str:
    """Return human-readable model version from file path."""
    name = os.path.basename(path).lower()
    if "v3" in name:
        return "v3"
    if "v2" in name:
        return "v2"
    return "custom"


def load_model(model_path: str | None = None):
    """
    Load the trained model onto CPU.

    Uses v3 if available, otherwise v2, unless model_path is specified.
    """
    path = model_path or resolve_model_path()
    ensure_model_file(path, MODEL_URL)

    device = torch.device("cpu")
    model = build_model(NUM_CLASSES)

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading model: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading model: %s", unexpected[:5])

    model.to(device)
    model.eval()
    version = get_model_version_label(path)
    logger.info("Model %s loaded from %s (CPU inference)", version, path)
    return model


def model_diagnostics(model_path: str | None = None) -> dict:
    """Diagnostics for /health endpoint."""
    path = model_path or resolve_model_path()
    info = get_model_file_info(path)
    info["model_version"] = get_model_version_label(path)
    info["path"] = path
    info["v3_exists"] = os.path.isfile(MODEL_V3_PATH)
    info["v2_exists"] = os.path.isfile(MODEL_V2_PATH)
    return info
