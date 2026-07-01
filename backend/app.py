"""
DENTRAT — Flask API Server
Serves frontend SPA + ML inference + auth + saved analyses + PDF reports.
"""
import logging
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, abort, jsonify, request, send_file, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

from auth import get_current_user_id, hash_password, login_user, logout_user, verify_password
from config import (
    ALLOWED_EXTENSIONS,
    CLASS_COLORS,
    CLASS_NAMES,
    DEBUG,
    DETECTABLE_CONDITIONS,
    HOST,
    MAX_UPLOAD_BYTES,
    MODEL_PATH,
    PORT,
    PROJECT_ROOT,
    SAVED_IMAGES_DIR,
    SECRET_KEY,
    UPLOAD_DIR,
)
from database import (
    create_user,
    delete_analysis,
    email_exists,
    get_analysis_by_id,
    get_saved_analyses,
    get_stats,
    get_user_by_id,
    get_user_by_login,
    init_db,
    save_analysis,
    save_contact_message,
    update_analysis_comment,
    username_exists,
)
from inference import run_inference
from image_utils import load_image_from_path
from model_loader import load_model, model_diagnostics
from pdf_generator import generate_analysis_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
CORS(app, supports_credentials=True)

model = None
model_error = None
_model_lock = threading.Lock()


def get_model():
    """Thread-safe lazy model loader — loads once on first analysis request."""
    global model, model_error
    if model is not None:
        return model
    with _model_lock:
        if model is not None:
            return model
        try:
            logger.info("Loading ML model (first request)...")
            model = load_model()
            model_error = None
            logger.info("Model loaded successfully")
        except Exception as exc:
            model_error = str(exc)
            logger.error("Model load failed: %s", exc)
            raise
    return model


try:
    init_db()
    os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Pre-load model at startup if possible; analysis will retry via get_model()
    try:
        model = load_model()
        logger.info("Server ready — model loaded at startup")
    except Exception as exc:
        model_error = str(exc)
        logger.warning("Model not loaded at startup (will retry on analyze): %s", exc)
except Exception as exc:
    model_error = str(exc)
    logger.error("Startup failed: %s", exc)

SPA_ROUTES = {
    "", "login", "signup", "dashboard", "results", "saved", "help", "contact",
}
STATIC_EXTENSIONS = {".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    """Decorator — return 401 if user not logged in."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not get_current_user_id():
            return jsonify({"error": "Unauthorized. Please log in."}), 401
        return f(*args, **kwargs)
    return decorated


def serve_frontend_file(path: str):
    safe_path = path.lstrip("/")
    full_path = os.path.join(FRONTEND_DIR, safe_path)
    if os.path.isfile(full_path):
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        return send_from_directory(directory, filename)
    return None


# ─── Auth ───

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    confirm = data.get("confirm_password") or ""
    organization = (data.get("organization") or "").strip() or None
    contact = (data.get("contact") or "").strip() or None

    if not all([full_name, email, username, password]):
        return jsonify({"error": "Full name, email, username, and password are required."}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    if username_exists(username):
        return jsonify({"error": "Username already taken."}), 409
    if email_exists(email):
        return jsonify({"error": "Email already registered."}), 409

    try:
        user = create_user(
            full_name=full_name,
            email=email,
            username=username,
            password_hash=hash_password(password),
            organization=organization,
            contact=contact,
        )
        return jsonify({"success": True, "message": "Account created. Please log in.", "user": user}), 201
    except Exception as exc:
        logger.exception("Signup failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    login_id = (data.get("username") or data.get("email") or "").strip()
    password = data.get("password") or ""

    if not login_id or not password:
        return jsonify({"error": "Username/email and password are required."}), 400

    user = get_user_by_login(login_id)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid username/email or password."}), 401

    login_user(user["id"], user["username"], user["full_name"])
    return jsonify(
        {
            "success": True,
            "user": {
                "id": user["id"],
                "full_name": user["full_name"],
                "username": user["username"],
                "email": user["email"],
                "organization": user.get("organization"),
            },
        }
    )


@app.route("/logout", methods=["POST"])
def logout():
    logout_user()
    return jsonify({"success": True})


@app.route("/me", methods=["GET"])
def me():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"logged_in": False}), 200
    user = get_user_by_id(user_id)
    return jsonify({"logged_in": True, "user": user})


# ─── Contact ───

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.route("/contact", methods=["POST"])
def contact():
    """Save a contact form submission (public — no login required)."""
    data = request.get_json(silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    message = (data.get("message") or "").strip()
    organization = (data.get("organization") or "").strip() or None
    phone = (data.get("phone") or "").strip() or None

    if not full_name:
        return jsonify({"error": "Full name is required."}), 400
    if not email:
        return jsonify({"error": "Email is required."}), 400
    if not _EMAIL_RE.match(email):
        return jsonify({"error": "Please enter a valid email address."}), 400
    if not message:
        return jsonify({"error": "Message is required."}), 400
    if len(message) > 5000:
        return jsonify({"error": "Message is too long (max 5000 characters)."}), 400

    try:
        record = save_contact_message(
            full_name=full_name,
            email=email,
            message=message,
            organization=organization,
            phone=phone,
        )
        return jsonify(
            {
                "success": True,
                "message": "Thank you! Your message has been received. We will get back to you soon.",
                "id": record["id"],
            }
        ), 201
    except Exception as exc:
        logger.exception("Contact form save failed")
        return jsonify({"error": "Could not save your message. Please try again later."}), 500


# ─── Analysis ───

def _find_temp_file(temp_id: str) -> str | None:
    """Locate uploaded image by temp_id prefix in UPLOAD_DIR."""
    if not temp_id:
        return None
    for name in os.listdir(UPLOAD_DIR):
        if name.startswith(f"{temp_id}_"):
            return os.path.join(UPLOAD_DIR, name)
    return None


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    """Upload X-ray, run model, return detections (does not save yet)."""
    try:
        active_model = get_model()
    except Exception:
        return jsonify({"error": "Model not loaded", "detail": model_error or "Check /health"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing image file."}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File too large (max 25 MB)."}), 400
    if len(file_bytes) == 0:
        return jsonify({"error": "Empty file."}), 400

    safe_name = secure_filename(file.filename)
    temp_id = uuid.uuid4().hex
    temp_path = os.path.join(UPLOAD_DIR, f"{temp_id}_{safe_name}")

    try:
        # Save to disk for later save-analysis step
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        del file_bytes  # free RAM before inference

        image = load_image_from_path(temp_path)
        img_w, img_h = image.size
        detections = run_inference(active_model, image)
        del image

        analysis_date = datetime.now(timezone.utc).isoformat()

        return jsonify(
            {
                "success": True,
                "temp_id": temp_id,
                "filename": safe_name,
                "image_width": img_w,
                "image_height": img_h,
                "analysis_date": analysis_date,
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
    except MemoryError:
        logger.exception("Out of memory during analysis")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify(
            {
                "error": "Server ran out of memory during analysis.",
                "detail": "Upgrade Railway RAM to at least 2 GB in Settings → Resources.",
            }
        ), 503
    except Exception as exc:
        logger.exception("Analyze failed")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": "Analysis failed", "detail": str(exc)}), 500


@app.route("/save-analysis", methods=["POST"])
@login_required
def save_analysis_route():
    """Save analysis with patient info to database."""
    data = request.get_json(silent=True) or {}
    user_id = get_current_user_id()

    temp_id = data.get("temp_id", "")
    detections = data.get("detections", [])
    analysis_date = data.get("analysis_date") or datetime.now(timezone.utc).isoformat()

    temp_path = _find_temp_file(temp_id)
    if not temp_path or not os.path.isfile(temp_path):
        return jsonify({"error": "No analysis image found. Please re-run analysis."}), 400

    # Move image to permanent saved storage
    ext = os.path.splitext(temp_path)[1] or ".jpg"
    permanent_name = f"{uuid.uuid4().hex}{ext}"
    permanent_path = os.path.join(SAVED_IMAGES_DIR, permanent_name)
    shutil.move(temp_path, permanent_path)

    try:
        saved = save_analysis(
            user_id=user_id,
            patient_name=(data.get("patient_name") or "").strip() or None,
            patient_contact=(data.get("patient_contact") or "").strip() or None,
            patient_email=(data.get("patient_email") or "").strip() or None,
            analysis_date=analysis_date,
            image_path=permanent_path,
            results=detections,
            comment=(data.get("comment") or "").strip() or None,
        )
        return jsonify({"success": True, "analysis": saved}), 201
    except Exception as exc:
        logger.exception("Save analysis failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/saved-analyses", methods=["GET"])
@login_required
def saved_analyses_list():
    user_id = get_current_user_id()
    return jsonify({"analyses": get_saved_analyses(user_id)})


@app.route("/analysis/<int:analysis_id>", methods=["GET"])
@login_required
def get_analysis(analysis_id):
    user_id = get_current_user_id()
    analysis = get_analysis_by_id(analysis_id, user_id)
    if not analysis:
        return jsonify({"error": "Analysis not found."}), 404
    return jsonify({"analysis": analysis})


@app.route("/analysis/<int:analysis_id>", methods=["DELETE"])
@login_required
def delete_analysis_route(analysis_id):
    user_id = get_current_user_id()
    if not delete_analysis(analysis_id, user_id):
        return jsonify({"error": "Analysis not found."}), 404
    return jsonify({"success": True})


@app.route("/analysis/<int:analysis_id>/comment", methods=["PUT"])
@login_required
def update_comment(analysis_id):
    user_id = get_current_user_id()
    data = request.get_json(silent=True) or {}
    comment = (data.get("comment") or "").strip()
    updated = update_analysis_comment(analysis_id, user_id, comment)
    if not updated:
        return jsonify({"error": "Analysis not found."}), 404
    return jsonify({"success": True, "analysis": updated})


@app.route("/analysis/<int:analysis_id>/pdf", methods=["GET"])
@login_required
def download_pdf(analysis_id):
    user_id = get_current_user_id()
    analysis = get_analysis_by_id(analysis_id, user_id)
    if not analysis:
        return jsonify({"error": "Analysis not found."}), 404

    try:
        pdf_bytes = generate_analysis_pdf(analysis)
        return send_file(
            __import__("io").BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"DENTRAT_Report_{analysis_id}.pdf",
        )
    except Exception as exc:
        logger.exception("PDF generation failed")
        return jsonify({"error": "PDF generation failed", "detail": str(exc)}), 500


# ─── Legacy / utility ───

@app.route("/health", methods=["GET"])
def health():
    file_info = model_diagnostics()
    return jsonify(
        {
            "status": "healthy" if model is not None else "degraded",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "model_version": file_info.get("model_version", "unknown"),
            "v3_available": file_info.get("v3_exists", False),
            "v2_available": file_info.get("v2_exists", False),
            "model_exists": file_info.get("exists", False),
            "model_file_size_mb": file_info.get("size_mb", 0),
            "model_file_valid": file_info.get("valid", False),
            "error": model_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classes": CLASS_NAMES,
        }
    )


@app.route("/stats", methods=["GET"])
def stats():
    data = get_stats()
    data["model_loaded"] = model is not None
    data["class_names"] = CLASS_NAMES
    data["conditions"] = DETECTABLE_CONDITIONS
    return jsonify(data)


# Keep /upload as alias for /analyze for backwards compatibility
@app.route("/upload", methods=["POST"])
@login_required
def upload_alias():
    return analyze()


# ─── Frontend SPA ───

@app.route("/")
@app.route("/login")
@app.route("/signup")
@app.route("/dashboard")
@app.route("/results")
@app.route("/saved")
@app.route("/help")
@app.route("/contact")
def serve_spa():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filepath>")
def serve_static(filepath):
    _, ext = os.path.splitext(filepath)
    if ext.lower() in STATIC_EXTENSIONS:
        response = serve_frontend_file(filepath)
        if response:
            return response
    if filepath in SPA_ROUTES or filepath.startswith("saved/"):
        return send_from_directory(FRONTEND_DIR, "index.html")
    abort(404)


if __name__ == "__main__":
    logger.info("Starting DENTRAT server on %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=DEBUG)
