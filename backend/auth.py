"""
User authentication helpers — bcrypt password hashing and session utilities.
"""
import bcrypt
from flask import session


def hash_password(password: str) -> str:
    """Hash a plain-text password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Return True if password matches the stored hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def login_user(user_id: int, username: str, full_name: str) -> None:
    """Store user info in Flask session cookie."""
    session["user_id"] = user_id
    session["username"] = username
    session["full_name"] = full_name
    session.permanent = True


def logout_user() -> None:
    """Clear session."""
    session.clear()


def get_current_user_id() -> int | None:
    return session.get("user_id")
