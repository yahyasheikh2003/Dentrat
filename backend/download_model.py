"""
Download and validate the dental model file for Railway deployment.

Google Drive share links often save HTML instead of the .pth file.
This module detects corrupt downloads and re-fetches using gdown.
"""
import logging
import os
import re
import urllib.request

logger = logging.getLogger(__name__)

# Real Faster R-CNN weights are typically 100MB+; HTML error pages are tiny
MIN_MODEL_BYTES = 5 * 1024 * 1024  # 5 MB


def extract_google_drive_id(url: str) -> str | None:
    """Extract file ID from common Google Drive URL formats."""
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/uc\?.*id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_valid_model_file(model_path: str) -> bool:
    """
    Return True if the file looks like a real PyTorch checkpoint (not HTML/JSON).
    """
    if not os.path.isfile(model_path):
        return False

    size = os.path.getsize(model_path)
    if size < MIN_MODEL_BYTES:
        logger.warning("Model file too small (%d bytes) — likely not a real .pth", size)
        return False

    with open(model_path, "rb") as f:
        header = f.read(256)

    # HTML / XML error pages from bad download links
    if header.lstrip().startswith(b"<") or header.lstrip().startswith(b"<!"):
        logger.warning("Model file is HTML, not PyTorch weights")
        return False

    # JSON error responses
    if header.lstrip().startswith(b"{") or header.lstrip().startswith(b"["):
        logger.warning("Model file is JSON, not PyTorch weights")
        return False

    return True


def download_from_google_drive(file_id: str, dest_path: str) -> None:
    """Download a Google Drive file using gdown (handles large files + confirm tokens)."""
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for Google Drive downloads. "
            "It should be installed via requirements.txt."
        ) from exc

    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info("Downloading from Google Drive (id=%s) ...", file_id)
    gdown.download(url, dest_path, quiet=False, fuzzy=True)


def download_from_url(url: str, dest_path: str) -> None:
    """Download model from a direct URL or Google Drive link."""
    drive_id = extract_google_drive_id(url)
    if drive_id:
        download_from_google_drive(drive_id, dest_path)
        return

    logger.info("Downloading model from URL: %s", url[:80])
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (DENTRAT/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = response.read()

    if data.lstrip().startswith(b"<") or len(data) < MIN_MODEL_BYTES:
        raise RuntimeError(
            "Download returned HTML or a tiny file — not a valid .pth. "
            "Use a Google Drive link or a direct file URL."
        )

    with open(dest_path, "wb") as f:
        f.write(data)


def ensure_model_file(model_path: str, model_url: str, force: bool = False) -> None:
    """
    Ensure a valid model file exists at model_path.

    Re-downloads if the file is missing, corrupt (HTML), or force=True.
    Set FORCE_MODEL_REDOWNLOAD=1 on Railway to replace a bad cached file.
    """
    force = force or os.environ.get("FORCE_MODEL_REDOWNLOAD", "0") == "1"

    if os.path.isfile(model_path):
        if force:
            logger.info("FORCE_MODEL_REDOWNLOAD: removing existing model file")
            os.remove(model_path)
        elif is_valid_model_file(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info("Valid model file found (%.1f MB) at %s", size_mb, model_path)
            return
        else:
            logger.warning("Corrupt model file at %s — deleting and re-downloading", model_path)
            os.remove(model_path)

    if not model_url:
        raise FileNotFoundError(
            f"No valid model at '{model_path}'. "
            "Set MODEL_URL in Railway Variables to a Google Drive link "
            "or direct download URL for dental_model_v2.pth."
        )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_from_url(model_url, model_path)

    if not is_valid_model_file(model_path):
        if os.path.isfile(model_path):
            os.remove(model_path)
        raise RuntimeError(
            "Download completed but file is not a valid PyTorch model. "
            "Check MODEL_URL — use Google Drive 'Anyone with link' sharing."
        )

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info("Model downloaded successfully (%.1f MB)", size_mb)


def get_model_file_info(model_path: str) -> dict:
    """Return diagnostic info for /health endpoint."""
    if not os.path.isfile(model_path):
        return {"exists": False, "size_mb": 0, "valid": False}

    size = os.path.getsize(model_path)
    valid = is_valid_model_file(model_path)
    preview = ""
    try:
        with open(model_path, "rb") as f:
            preview = f.read(20).decode("utf-8", errors="replace")
    except Exception:
        preview = "?"

    return {
        "exists": True,
        "size_mb": round(size / (1024 * 1024), 2),
        "valid": valid,
        "header_preview": preview[:20] if not valid else "ok",
    }
