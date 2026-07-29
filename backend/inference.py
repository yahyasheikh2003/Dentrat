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

from config import CLASS_NAMES, CONFIDENCE_THRESHOLD, EXCLUDED_CLASS_IDS, IMAGE_SIZE

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


def get_severity(confidence: float) -> str:
    """Map model confidence to clinical severity tier."""
    if confidence >= 0.85:
        return "High Severity"
    if confidence >= 0.65:
        return "Moderate Severity"
    return "Low Severity"


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
            if class_id not in CLASS_NAMES or class_id in EXCLUDED_CLASS_IDS:
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
                }
            )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        logger.info("Found %d detections above threshold %.2f", len(detections), confidence_threshold)
        return detections
