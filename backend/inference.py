"""
Run Faster R-CNN inference on dental X-ray images — optimized for low-RAM Railway CPU.
"""
import gc
import logging
import os
import threading
from typing import Any

# Limit CPU thread count BEFORE importing torch (reduces memory on small VMs)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from PIL import Image
from torchvision import transforms

from config import CLASS_NAMES, CONFIDENCE_THRESHOLD, IMAGE_SIZE

logger = logging.getLogger(__name__)
torch.set_num_threads(1)

# Only one inference at a time — prevents parallel requests doubling RAM usage
_inference_lock = threading.Lock()

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _describe_location(bbox: list[float], img_w: int, img_h: int) -> str:
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    vertical = "Upper" if cy < img_h / 2 else "Lower"
    horizontal = "Left" if cx < img_w / 2 else "Right"
    if abs(cx - img_w / 2) < img_w * 0.15 and abs(cy - img_h / 2) < img_h * 0.15:
        return "Center"
    return f"{vertical} {horizontal}"


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    original_size = image.size
    tensor = TRANSFORM(image)
    return tensor.unsqueeze(0), original_size


def run_inference(
    model,
    image: Image.Image,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Run detection with memory-safe locking for Railway deployment."""
    with _inference_lock:
        tensor, (orig_w, orig_h) = preprocess_image(image)

        try:
            with torch.inference_mode():
                outputs = model(tensor)

            output = outputs[0]
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
        finally:
            # Free tensor memory immediately
            del tensor
            gc.collect()

        scale_x = orig_w / IMAGE_SIZE
        scale_y = orig_h / IMAGE_SIZE

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            if score < confidence_threshold:
                continue
            class_id = int(label)
            if class_id not in CLASS_NAMES:
                continue
            x1, y1, x2, y2 = box
            x1 = float(x1 * scale_x)
            y1 = float(y1 * scale_y)
            x2 = float(x2 * scale_x)
            y2 = float(y2 * scale_y)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append(
                {
                    "class_id": class_id,
                    "class": CLASS_NAMES[class_id],
                    "bbox": [round(v, 2) for v in bbox],
                    "confidence": round(float(score), 4),
                    "location": _describe_location(bbox, orig_w, orig_h),
                }
            )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        logger.info("Found %d detections above threshold %.2f", len(detections), confidence_threshold)
        return detections
