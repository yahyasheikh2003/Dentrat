"""
Load the Faster R-CNN dental anomaly detector for CPU inference.

On Railway, set MODEL_URL to auto-download the .pth file on first startup.
"""
import logging
import os
import urllib.request

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import MODEL_PATH, MODEL_URL, NUM_CLASSES, PROJECT_ROOT

logger = logging.getLogger(__name__)


def build_model(num_classes: int = NUM_CLASSES):
    """Build Faster R-CNN with ResNet50-FPN backbone."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def ensure_model_file(model_path: str = MODEL_PATH) -> None:
    """
    Ensure the model file exists locally.

    If missing and MODEL_URL is set (Railway env var), download it automatically.
    """
    if os.path.isfile(model_path):
        logger.info("Model file found at %s", model_path)
        return

    if not MODEL_URL:
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Either commit is wrong — place dental_model_v2.pth in models/, "
            "or set MODEL_URL in Railway Variables to a direct download link."
        )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    logger.info("Downloading model from MODEL_URL to %s ...", model_path)

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to download model from MODEL_URL: {exc}") from exc

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info("Model downloaded successfully (%.1f MB)", size_mb)


def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained model onto CPU.

    Returns:
        model: Eval-mode Faster R-CNN on CPU
    """
    ensure_model_file(model_path)

    device = torch.device("cpu")
    model = build_model(NUM_CLASSES)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

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
    logger.info("Model loaded from %s (CPU inference)", model_path)
    return model
