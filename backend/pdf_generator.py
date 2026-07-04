"""
Generate professional PDF reports for DENTRAT analyses using ReportLab.
"""
import io
import os
import re
from datetime import datetime

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import CLASS_COLORS

NAVY_DARK = colors.HexColor("#0d1b4a")
NAVY_PRIMARY = colors.HexColor("#1a2a6c")
NAVY_MID = colors.HexColor("#2a3a7c")
NAVY_TINT = colors.HexColor("#eef2fb")
TEXT_MUTED = colors.HexColor("#5a6888")


def _safe(value, fallback="Not provided") -> str:
    if value is None or str(value).strip() in ("", "—", "-"):
        return fallback
    return str(value).strip()


def _pdf_tooth_cell(det: dict) -> str:
    """Compact tooth label for PDF table cells (avoids overflow)."""
    tooth = det.get("tooth") or ""
    match = re.search(r"#(\d+)", tooth)
    num = f"#{match.group(1)}" if match else ""
    region = det.get("location") or ""
    if tooth and "(" in tooth:
        region = tooth.split("(")[0].strip()
    if num and region:
        return f"{num}<br/><font size='7' color='#5a6888'>{region}</font>"
    if num:
        return num
    return _safe(region, "See image")


def _draw_annotated_image(image_path: str, detections: list[dict], max_width: int = 500) -> io.BytesIO | None:
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
        color_hex = det.get("color") or CLASS_COLORS.get(det.get("class_id"), "#1a2a6c")
        draw.rectangle([x, y, x + bw, y + bh], outline=color_hex, width=3)
        label = f"{det.get('class', '?')} {int(det.get('confidence', 0) * 100)}%"
        draw.rectangle([x, max(y - 18, 0), x + len(label) * 7, max(y - 18, 0) + 16], fill=color_hex)
        draw.text((x + 2, max(y - 16, 2)), label, fill="white")

    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def generate_analysis_pdf(analysis: dict) -> bytes:
    buffer = io.BytesIO()
    page_w, page_h = A4
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=45,
        title="DENTRAT AI Diagnostic Report",
    )

    styles = getSampleStyleSheet()
    brand_title = ParagraphStyle(
        "BrandTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=NAVY_PRIMARY,
        spaceAfter=2,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    brand_sub = ParagraphStyle(
        "BrandSub",
        parent=styles["Normal"],
        fontSize=11,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    section_head = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=NAVY_DARK,
        spaceBefore=14,
        spaceAfter=8,
        fontName="Helvetica-Bold",
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=colors.black,
        leading=13,
    )
    cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
    )
    cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=cell,
        fontName="Helvetica-Bold",
    )
    cell_label = ParagraphStyle(
        "TableCellLabel",
        parent=cell,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    cell_header = ParagraphStyle(
        "TableHeader",
        parent=cell,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    footer = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        leading=11,
    )

    story = []

    # Header block
    header_table = Table(
        [[Paragraph("DENTRAT", brand_title)], [Paragraph("AI Diagnostic Report", brand_sub)]],
        colWidths=[page_w - 90],
    )
    header_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NAVY_TINT),
            ("BOX", (0, 0), (-1, -1), 0.5, NAVY_MID),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 12))

    # Patient info
    story.append(Paragraph("Patient Information", section_head))
    analysis_date = analysis.get("analysis_date", "")
    try:
        dt = datetime.fromisoformat(analysis_date.replace("Z", "+00:00"))
        formatted_date = dt.strftime("%B %d, %Y at %H:%M UTC")
    except Exception:
        formatted_date = _safe(analysis_date)

    patient_rows = [
        [Paragraph("Patient Name", cell_label), Paragraph(_safe(analysis.get("patient_name")), cell)],
        [Paragraph("Contact", cell_label), Paragraph(_safe(analysis.get("patient_contact")), cell)],
        [Paragraph("Email", cell_label), Paragraph(_safe(analysis.get("patient_email")), cell)],
        [Paragraph("Analysis Date", cell_label), Paragraph(formatted_date, cell)],
    ]
    patient_table = Table(patient_rows, colWidths=[1.55 * inch, 4.45 * inch])
    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), NAVY_MID),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8deef")),
            ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, NAVY_TINT]),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )
    story.append(patient_table)
    story.append(Spacer(1, 14))

    # Findings table
    detections = analysis.get("detections", [])
    count = len(detections)
    story.append(Paragraph(f"Radiographic Findings ({count} detected)", section_head))

    if detections:
        headers = [
            Paragraph("#", cell_header),
            Paragraph("Condition", cell_header),
            Paragraph("Tooth / Region", cell_header),
            Paragraph("Severity", cell_header),
            Paragraph("Confidence", cell_header),
        ]
        table_data = [headers]
        for i, det in enumerate(detections, 1):
            conf = det.get("confidence", 0)
            table_data.append([
                Paragraph(str(i), cell),
                Paragraph(_safe(det.get("class"), "Unknown"), cell),
                Paragraph(_pdf_tooth_cell(det), cell),
                Paragraph(_safe(det.get("severity"), "Not assessed"), cell),
                Paragraph(f"{conf * 100:.1f}%", cell),
            ])

        col_widths = [0.32 * inch, 1.45 * inch, 1.55 * inch, 1.05 * inch, 0.85 * inch]
        results_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        results_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY_PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8deef")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, NAVY_TINT]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(results_table)
    else:
        story.append(
            Paragraph(
                "No dental anomalies were detected above the confidence threshold.",
                body,
            )
        )

    story.append(Spacer(1, 14))

    img_buf = _draw_annotated_image(analysis.get("image_path", ""), detections)
    if img_buf:
        story.append(Paragraph("Annotated Radiograph", section_head))
        img = Image.open(img_buf)
        img_w, img_h = img.size
        display_w = 5.8 * inch
        display_h = display_w * (img_h / max(img_w, 1))
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=display_w, height=display_h)
        rl_img.hAlign = "CENTER"
        story.append(rl_img)

    if analysis.get("comment"):
        story.append(Spacer(1, 10))
        story.append(Paragraph("Clinical Notes", section_head))
        story.append(Paragraph(_safe(analysis.get("comment")), body))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d8deef")))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "This is an AI generated report. Please consult your doctor before making clinical decisions.",
            footer,
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Generated by DENTRAT AI &nbsp;&bull;&nbsp; Confidential Medical Report &nbsp;&bull;&nbsp; &copy; 2026 DENTRAT",
            footer,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
