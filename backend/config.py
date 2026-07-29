"""
Configuration for DENTRAT backend (Railway / local).
"""
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_V3_PATH = os.path.join(MODELS_DIR, "dental_model_v3.pth")
MODEL_V2_PATH = os.path.join(MODELS_DIR, "dental_model_v2.pth")


def resolve_model_path() -> str:
    """
    Pick the best available model file.
    Priority: MODEL_PATH env → v2 → v3 fallback
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return env_path
    if os.path.isfile(MODEL_V2_PATH):
        return MODEL_V2_PATH
    if os.path.isfile(MODEL_V3_PATH):
        return MODEL_V3_PATH
    return MODEL_V2_PATH


MODEL_PATH = resolve_model_path()
MODEL_URL = os.environ.get("MODEL_URL", "")


def _resolve_data_dir() -> str:
    """Persistent storage for DB and saved images (use Railway Volume at /data)."""
    env_dir = os.environ.get("DATA_DIR")
    if env_dir:
        return env_dir
    if os.path.isdir("/data"):
        return "/data"
    return BACKEND_DIR


DATA_DIR = _resolve_data_dir()
DATABASE_PATH = os.path.join(DATA_DIR, "dental_history.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/dental_uploads")
SAVED_IMAGES_DIR = os.environ.get(
    "SAVED_IMAGES_DIR",
    os.path.join(DATA_DIR, "saved_images"),
)

SECRET_KEY = os.environ.get("SECRET_KEY", "dentrat-change-this-in-production")

# Contact: public address shown on website; notifications go to admin inbox
CONTACT_DISPLAY_EMAIL = "contact.dentrat@gmail.com"
CONTACT_NOTIFY_EMAIL = os.environ.get("CONTACT_NOTIFY_EMAIL", "yahyasheikh2003@gmail.com")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", CONTACT_DISPLAY_EMAIL)


def smtp_configured() -> bool:
    return bool(SMTP_USER and SMTP_PASSWORD)

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

# Classes detected by the model but hidden from users (dashboard, results, PDF)
EXCLUDED_CLASS_IDS = {6}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048

DETECTABLE_CONDITIONS = [
    {"id": 1, "name": "Caries & Cavities", "desc": "Tooth decay and cavity formations"},
    {"id": 2, "name": "Impacted Teeth", "desc": "Impacted teeth and positioning issues"},
    {"id": 3, "name": "Broken Crown/Root", "desc": "Damaged crowns and broken structures"},
    {"id": 4, "name": "Infection", "desc": "Dental infections and abscesses"},
    {"id": 5, "name": "Fractured Teeth", "desc": "Tooth fractures and structural cracks"},
    {"id": 7, "name": "Other Abnormalities", "desc": "Other radiographic abnormalities"},
]
