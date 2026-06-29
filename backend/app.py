"""
Dental Anomaly Detection — Flask API Server (DENTRAT)

Serves the DENTRAT frontend SPA and ML inference API.
"""
import logging
import os
import uuid
from datetime import datetime, timezone

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    CLASS_COLORS,
    CLASS_NAMES,
    DEBUG,
    HOST,
    MAX_UPLOAD_BYTES,
    MODEL_PATH,
    PORT,
    PROJECT_ROOT,
    UPLOAD_DIR,
)
from database import get_history, get_stats, init_db, save_detections, save_upload
from inference import run_inference
from model_loader import load_model, model_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

model = None
model_error = None

try:
    init_db()
    model = load_model()
    logger.info("Server ready — model loaded successfully")
except Exception as exc:
    model_error = str(exc)
    logger.error("Failed to load model: %s", exc)

# SPA routes served by index.html
SPA_ROUTES = {"", "login", "results", "help"}

# Static asset extensions served as files
STATIC_EXTENSIONS = {".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def serve_frontend_file(path: str):
    """Serve a static file from frontend/ if it exists."""
    safe_path = path.lstrip("/")
    full_path = os.path.join(FRONTEND_DIR, safe_path)
    if os.path.isfile(full_path):
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        return send_from_directory(directory, filename)
    return None


# ─── API routes (must be registered before catch-all) ───

@app.route("/health", methods=["GET"])
def health():
    file_info = model_diagnostics()
    return jsonify(
        {
            "status": "healthy" if model is not None else "degraded",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "model_exists": file_info.get("exists", False),
            "model_file_size_mb": file_info.get("size_mb", 0),
            "model_file_valid": file_info.get("valid", False),
            "model_header_preview": file_info.get("header_preview", ""),
            "model_url_set": bool(os.environ.get("MODEL_URL", "")),
            "error": model_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classes": CLASS_NAMES,
            "fix_hint": (
                "Set MODEL_URL to your Google Drive link, add FORCE_MODEL_REDOWNLOAD=1, redeploy."
                if not model and file_info.get("exists") and not file_info.get("valid")
                else None
            ),
        }
    )


@app.route("/stats", methods=["GET"])
def stats():
    try:
        data = get_stats()
        data["model_loaded"] = model is not None
        data["class_names"] = CLASS_NAMES
        data["class_colors"] = CLASS_COLORS
        return jsonify(data)
    except Exception as exc:
        logger.exception("Stats error")
        return jsonify({"error": str(exc)}), 500


@app.route("/history", methods=["GET"])
def history():
    try:
        limit = request.args.get("limit", 50, type=int)
        return jsonify({"history": get_history(limit=limit)})
    except Exception as exc:
        logger.exception("History error")
        return jsonify({"error": str(exc)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    if model is None:
        return jsonify(
            {"error": "Model not loaded", "detail": model_error or "Check /health"}
        ), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type", "allowed": sorted(ALLOWED_EXTENSIONS)}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB."}), 400
    if len(file_bytes) == 0:
        return jsonify({"error": "Empty file"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        image = Image.open(save_path)
        image.verify()
        image = Image.open(save_path)

        detections = run_inference(model, image)
        image_id = save_upload(filename=safe_name, file_path=save_path)
        save_detections(image_id, detections)

        return jsonify(
            {
                "success": True,
                "image_id": image_id,
                "filename": safe_name,
                "image_width": image.size[0],
                "image_height": image.size[1],
                "detection_count": len(detections),
                "detections": [
                    {
                        "class_id": d["class_id"],
                        "class": d["class"],
                        "bbox": d["bbox"],
                        "confidence": d["confidence"],
                        "location": d["location"],
                        "color": CLASS_COLORS.get(d["class_id"], "#FFFFFF"),
                    }
                    for d in detections
                ],
            }
        )
    except Exception as exc:
        logger.exception("Upload/inference failed")
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": "Processing failed", "detail": str(exc)}), 500


# ─── Frontend SPA routes ───

@app.route("/")
@app.route("/login")
@app.route("/results")
@app.route("/help")
def serve_spa():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filepath>")
def serve_static(filepath):
    """Serve CSS/JS assets or fall back to SPA index."""
    _, ext = os.path.splitext(filepath)
    if ext.lower() in STATIC_EXTENSIONS:
        response = serve_frontend_file(filepath)
        if response:
            return response
    if filepath in SPA_ROUTES:
        return send_from_directory(FRONTEND_DIR, "index.html")
    abort(404)


if __name__ == "__main__":
    logger.info("Starting DENTRAT server on %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=DEBUG)
