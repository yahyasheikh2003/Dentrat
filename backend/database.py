"""
SQLite database — users, saved analyses, and legacy upload tables.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from config import DATABASE_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                organization TEXT,
                contact TEXT,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS saved_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_name TEXT,
                patient_contact TEXT,
                patient_email TEXT,
                analysis_date TEXT NOT NULL,
                image_path TEXT,
                results_json TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                file_path TEXT
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                bbox TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(id)
            );

            CREATE TABLE IF NOT EXISTS contact_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                organization TEXT,
                email TEXT NOT NULL,
                phone TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'unread'
            );
            """
        )


# ─── Users ───

def create_user(
    full_name: str,
    email: str,
    username: str,
    password_hash: str,
    organization: str | None = None,
    contact: str | None = None,
) -> dict[str, Any]:
    created_at = _now_iso()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (full_name, organization, contact, email, username, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (full_name, organization, contact, email, username, password_hash, created_at),
        )
        return {
            "id": cursor.lastrowid,
            "full_name": full_name,
            "organization": organization,
            "contact": contact,
            "email": email,
            "username": username,
            "created_at": created_at,
        }


def get_user_by_username(username: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_login(login: str) -> dict | None:
    """Find user by username OR email."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM users
            WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE
            """,
            (login, login),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        u = dict(row)
        u.pop("password_hash", None)
        return u


def username_exists(username: str) -> bool:
    return get_user_by_username(username) is not None


def email_exists(email: str) -> bool:
    return get_user_by_email(email) is not None


# ─── Saved analyses ───

def save_analysis(
    user_id: int,
    patient_name: str | None,
    patient_contact: str | None,
    patient_email: str | None,
    analysis_date: str,
    image_path: str,
    results: list[dict],
    comment: str | None = None,
) -> dict[str, Any]:
    created_at = _now_iso()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO saved_analyses
                (user_id, patient_name, patient_contact, patient_email,
                 analysis_date, image_path, results_json, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                patient_name,
                patient_contact,
                patient_email,
                analysis_date,
                image_path,
                json.dumps(results),
                comment,
                created_at,
            ),
        )
        return get_analysis_by_id(cursor.lastrowid, user_id)


def get_saved_analyses(user_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, patient_name, patient_contact, patient_email,
                   analysis_date, image_path, results_json, comment, created_at
            FROM saved_analyses
            WHERE user_id = ?
            ORDER BY analysis_date DESC
            """,
            (user_id,),
        ).fetchall()

    results = []
    for row in rows:
        detections = json.loads(row["results_json"])
        results.append(
            {
                "id": row["id"],
                "patient_name": row["patient_name"] or "Unnamed Patient",
                "patient_contact": row["patient_contact"],
                "patient_email": row["patient_email"],
                "analysis_date": row["analysis_date"],
                "detection_count": len(detections),
                "comment": row["comment"],
                "created_at": row["created_at"],
            }
        )
    return results


def get_analysis_by_id(analysis_id: int, user_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM saved_analyses WHERE id = ? AND user_id = ?
            """,
            (analysis_id, user_id),
        ).fetchone()
        if not row:
            return None
        detections = json.loads(row["results_json"])
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "patient_name": row["patient_name"],
            "patient_contact": row["patient_contact"],
            "patient_email": row["patient_email"],
            "analysis_date": row["analysis_date"],
            "image_path": row["image_path"],
            "detections": detections,
            "detection_count": len(detections),
            "comment": row["comment"],
            "created_at": row["created_at"],
        }


def delete_analysis(analysis_id: int, user_id: int) -> bool:
    analysis = get_analysis_by_id(analysis_id, user_id)
    if not analysis:
        return False
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM saved_analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        )
    # Remove image file if it exists
    path = analysis.get("image_path")
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return True


def update_analysis_comment(analysis_id: int, user_id: int, comment: str) -> dict | None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE saved_analyses SET comment = ?
            WHERE id = ? AND user_id = ?
            """,
            (comment, analysis_id, user_id),
        )
    return get_analysis_by_id(analysis_id, user_id)


# ─── Contact messages ───

def save_contact_message(
    full_name: str,
    email: str,
    message: str,
    organization: str | None = None,
    phone: str | None = None,
) -> dict[str, Any]:
    created_at = _now_iso()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO contact_messages
                (full_name, organization, email, phone, message, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, 'unread')
            """,
            (full_name, organization, email, phone, message, created_at),
        )
        return {
            "id": cursor.lastrowid,
            "full_name": full_name,
            "organization": organization,
            "email": email,
            "phone": phone,
            "message": message,
            "created_at": created_at,
            "status": "unread",
        }


# ─── Legacy stats (dashboard) ───

def get_stats() -> dict[str, Any]:
    with get_connection() as conn:
        total_analyses = conn.execute("SELECT COUNT(*) FROM saved_analyses").fetchone()[0]
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return {
        "total_analyses": total_analyses,
        "total_users": total_users,
    }
