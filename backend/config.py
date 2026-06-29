"""
Configuration for DENTRAT backend (Railway / local).
"""
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "dental_model_v2.pth"),
)
MODEL_URL = os.environ.get("MODEL_URL", "")

DATABASE_PATH = os.path.join(BACKEND_DIR, "dental_history.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/dental_uploads")
SAVED_IMAGES_DIR = os.environ.get(
    "SAVED_IMAGES_DIR",
    os.path.join(BACKEND_DIR, "saved_images"),
)

SECRET_KEY = os.environ.get("SECRET_KEY", "dentrat-change-this-in-production")

IMAGE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
NUM_CLASSES = 8

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

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_IMAGE_DIMENSION = 2048  # Downscale larger X-rays before inference (saves RAM)

# Conditions displayed in UI (informational only — no selection required)
DETECTABLE_CONDITIONS = [
    {"id": 1, "name": "Caries & Cavities", "desc": "Tooth decay and cavity formations"},
    {"id": 2, "name": "Impacted Teeth", "desc": "Impacted teeth and positioning issues"},
    {"id": 6, "name": "Periodontal Bone Loss", "desc": "Bone density and periodontal loss"},
    {"id": 3, "name": "Broken Crown/Root", "desc": "Damaged crowns and broken structures"},
    {"id": 4, "name": "Infection", "desc": "Dental infections and abscesses"},
    {"id": 5, "name": "Fractured Teeth", "desc": "Tooth fractures and structural cracks"},
    {"id": 7, "name": "Other Abnormalities", "desc": "Other radiographic abnormalities"},
]
