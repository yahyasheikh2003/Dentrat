"""
Send contact form notifications via SMTP (Gmail-compatible).
"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import (
    CONTACT_DISPLAY_EMAIL,
    CONTACT_NOTIFY_EMAIL,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
    smtp_configured,
)

logger = logging.getLogger(__name__)


def send_contact_notification(record: dict) -> bool:
    """
    Email contact form submission to the admin inbox.
    Returns True if sent, False if SMTP is not configured or send failed.
    """
    if not smtp_configured():
        logger.warning(
            "SMTP not configured (set SMTP_USER and SMTP_PASSWORD). "
            "Contact message saved to database only."
        )
        return False

    subject = f"DENTRAT Contact: {record.get('full_name', 'New message')}"
    org_line = f"Organization: {record['organization']}\n" if record.get("organization") else ""
    phone_line = f"Phone: {record['phone']}\n" if record.get("phone") else ""

    body = (
        f"New message from the DENTRAT contact form\n\n"
        f"Name: {record.get('full_name')}\n"
        f"{org_line}"
        f"Email: {record.get('email')}\n"
        f"{phone_line}\n"
        f"Message:\n{record.get('message')}\n\n"
        f"Reply directly to: {record.get('email')}\n"
        f"Received via {CONTACT_DISPLAY_EMAIL}"
    )

    msg = MIMEMultipart()
    msg["From"] = f"DENTRAT <{SMTP_FROM}>"
    msg["To"] = CONTACT_NOTIFY_EMAIL
    msg["Reply-To"] = record.get("email", CONTACT_DISPLAY_EMAIL)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [CONTACT_NOTIFY_EMAIL], msg.as_string())
        logger.info("Contact notification sent to %s", CONTACT_NOTIFY_EMAIL)
        return True
    except Exception:
        logger.exception("Failed to send contact email to %s", CONTACT_NOTIFY_EMAIL)
        return False
