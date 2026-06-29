"""
Configuration for the Dental Anomaly Detection backend.

Paths resolve relative to the project root for local dev, GitHub, and Railway.
"""
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Model — place dental_model_v2.pth here, or set MODEL_URL on Railway
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "dental_model_v2.pth"),
)
MODEL_URL = os.environ.get("MODEL_URL", "")

# SQLite database for upload history
DATABASE_PATH = os.path.join(BACKEND_DIR, "dental_history.db")

# Temporary upload storage (Railway/Linux: /tmp is fine)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/dental_uploads")

# Inference settings
IMAGE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
NUM_CLASSES = 8  # background (0) + 7 anomaly classes (1-7)

CLASS_NAMES = {
    1: "Caries",
    2: "Impacted Teeth",
    3: "Broken Down Crown/Root",
    4: "Infection",
    5: "Fractured Teeth",
    6: "Periodontal Bone Loss",
    7: "Other Abnormalities",
}

CLASS_COLORS = {
    1: "#FF4444",
    2: "#FF8800",
    3: "#FFCC00",
    4: "#44AA44",
    5: "#4488FF",
    6: "#AA44FF",
    7: "#FF44AA",
}

# Flask — Railway injects PORT automatically
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
