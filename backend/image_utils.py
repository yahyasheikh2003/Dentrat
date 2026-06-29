"""
Safe image loading and resizing before inference (reduces RAM spikes on Railway).
"""
import io
import logging

from PIL import Image

from config import MAX_IMAGE_DIMENSION

logger = logging.getLogger(__name__)

# Allow large dental X-rays but cap decompression bombs
Image.MAX_IMAGE_PIXELS = 40_000_000


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Load RGB image from bytes and downscale if very large."""
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")
    return _resize_if_needed(img)


def load_image_from_path(path: str) -> Image.Image:
    """Load RGB image from disk and downscale if very large."""
    img = Image.open(path)
    img = img.convert("RGB")
    return _resize_if_needed(img)


def _resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_dim = max(w, h)
    if max_dim > MAX_IMAGE_DIMENSION:
        ratio = MAX_IMAGE_DIMENSION / max_dim
        new_size = (int(w * ratio), int(h * ratio))
        logger.info("Resizing image from %dx%d to %dx%d", w, h, new_size[0], new_size[1])
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img
