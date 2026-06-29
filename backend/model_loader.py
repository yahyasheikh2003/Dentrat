"""
Load the Faster R-CNN dental anomaly detector for CPU inference.

On Railway, set S3 bucket credentials to auto-download the .pth file on first startup.
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


def download_from_s3(model_path: str) -> None:
    """Download model from Railway S3 bucket if credentials are available."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.warning("boto3 not available, skipping S3 download")
        return

    bucket_name = os.environ.get("BUCKET")
    access_key = os.environ.get("ACCESS_KEY_ID")
    secret_key = os.environ.get("SECRET_ACCESS_KEY")
    endpoint = os.environ.get("ENDPOINT")
    region = os.environ.get("REGION", "auto")

    if not all([bucket_name, access_key, secret_key, endpoint]):
        logger.info("S3 credentials not fully configured, skipping S3 download")
        return

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        # List objects to find the model file
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        model_key = None

        if "Contents" in response:
            for obj in response["Contents"]:
                if obj["Key"].endswith(".pth"):
                    model_key = obj["Key"]
                    break

        if not model_key:
            logger.warning("No .pth file found in S3 bucket")
            return

        logger.info("Downloading model from S3: %s/%s", bucket_name, model_key)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        s3_client.download_file(bucket_name, model_key, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info("Model downloaded successfully from S3 (%.1f MB)", size_mb)

    except ClientError as exc:
        logger.error("Failed to download from S3: %s", exc)
        raise RuntimeError(f"Failed to download model from S3: {exc}") from exc
    except Exception as exc:
        logger.error("Unexpected error downloading from S3: %s", exc)
        raise


def ensure_model_file(model_path: str = MODEL_PATH) -> None:
    """
    Ensure the model file exists locally.

    Try in order:
    1. Check if file exists locally
    2. Download from S3 bucket (Railway)
    3. Download from MODEL_URL if set
    """
    if os.path.isfile(model_path):
        logger.info("Model file found at %s", model_path)
        return

    # Try S3 first
    try:
        download_from_s3(model_path)
        if os.path.isfile(model_path):
            return
    except Exception as exc:
        logger.warning("S3 download failed: %s", exc)

    # Fall back to MODEL_URL
    if not MODEL_URL:
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Either: (1) commit the model file to models/, "
            "(2) set S3 bucket credentials (BUCKET, ACCESS_KEY_ID, SECRET_ACCESS_KEY, ENDPOINT), "
            "or (3) set MODEL_URL in Railway Variables to a direct download link."
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

