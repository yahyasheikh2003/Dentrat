"""
Generate professional PDF reports for DENTRAT analyses using ReportLab.
"""
import io
import json
import os
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import CLASS_COLORS

# Navy brand colors
NAVY_DARK = colors.HexColor("#0d1b4a")
NAVY_PRIMARY = colors.HexColor("#1a2a6c")
NAVY_LIGHT = colors.HexColor("#2a3a7c")


def _draw_annotated_image(image_path: str, detections: list[dict], max_width: int = 480) -> io.BytesIO | None:
    """Draw bounding boxes on the X-ray and return PNG bytes."""
    if not image_path or not os.path.isfile(image_path):
        return None

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, bw, bh = bbox
        color_hex = det.get("color") or CLASS_COLORS.get(det.get("class_id"), "#FF0000")
        # Draw rectangle
        draw.rectangle([x, y, x + bw, y + bh], outline=color_hex, width=3)
        label = f"{det.get('class', '?')} {int(det.get('confidence', 0) * 100)}%"
        draw.rectangle([x, max(y - 18, 0), x + len(label) * 7, max(y - 18, 0) + 16], fill=color_hex)
        draw.text((x + 2, max(y - 16, 2)), label, fill="white")

    # Scale down for PDF
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def generate_analysis_pdf(analysis: dict) -> bytes:
    """
    Build a PDF report for a saved analysis.

    Args:
        analysis: dict from get_analysis_by_id()

    Returns:
        PDF file as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
        title="DENTRAT Analysis Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DentratTitle",
        parent=styles["Heading1"],
        fontSize=22,
        textColor=NAVY_PRIMARY,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "DentratSub",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=NAVY_DARK,
        spaceBefore=16,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=4,
    )

    story = []

    # Header
    story.append(Paragraph("DENTRAT", title_style))
    story.append(Paragraph("Dental Radiography Analysis Tool — AI Diagnostic Report", subtitle_style))

    # Patient information
    story.append(Paragraph("Patient Information", heading_style))
    analysis_date = analysis.get("analysis_date", "")
    try:
        dt = datetime.fromisoformat(analysis_date.replace("Z", "+00:00"))
        formatted_date = dt.strftime("%B %d, %Y at %H:%M UTC")
    except Exception:
        formatted_date = analysis_date

    patient_data = [
        ["Patient Name", analysis.get("patient_name") or "—"],
        ["Contact", analysis.get("patient_contact") or "—"],
        ["Email", analysis.get("patient_email") or "—"],
        ["Analysis Date", formatted_date],
    ]
    patient_table = Table(patient_data, colWidths=[2 * inch, 4 * inch])
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), NAVY_LIGHT),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(patient_table)
    story.append(Spacer(1, 16))

    # Detection results
    detections = analysis.get("detections", [])
    story.append(Paragraph(f"Analysis Results — {len(detections)} Finding(s)", heading_style))

    if detections:
        table_data = [["#", "Condition", "Confidence", "Location"]]
        for i, det in enumerate(detections, 1):
            table_data.append(
                [
                    str(i),
                    det.get("class", "Unknown"),
                    f"{det.get('confidence', 0) * 100:.1f}%",
                    det.get("location", "—"),
                ]
            )
        results_table = Table(table_data, colWidths=[0.4 * inch, 2.5 * inch, 1.2 * inch, 1.9 * inch])
        results_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY_PRIMARY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(results_table)
    else:
        story.append(Paragraph("No dental anomalies detected above the confidence threshold.", body_style))

    story.append(Spacer(1, 16))

    # Annotated X-ray image
    img_buf = _draw_annotated_image(analysis.get("image_path", ""), detections)
    if img_buf:
        story.append(Paragraph("Annotated X-Ray", heading_style))
        rl_img = RLImage(img_buf, width=5.5 * inch, height=3.5 * inch)
        rl_img.hAlign = "CENTER"
        story.append(rl_img)

    # Comment
    if analysis.get("comment"):
        story.append(Spacer(1, 12))
        story.append(Paragraph("Clinical Notes", heading_style))
        story.append(Paragraph(analysis["comment"], body_style))

    # Footer
    story.append(Spacer(1, 24))
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )
    story.append(Paragraph("Generated by DENTRAT AI — Confidential Medical Report", footer_style))
    story.append(Paragraph("© 2026 DENTRAT. HIPAA Compliant • For clinical assistance only.", footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
