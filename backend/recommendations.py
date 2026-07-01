"""
Clinical descriptions, recommendations, and detection enrichment for DENTRAT.
"""
from typing import Any

from inference import estimate_tooth, get_severity

WEBSITE_FOOTER = (
    "*This AI-generated report should be reviewed by a licensed dental "
    "professional before making clinical decisions."
)

CLASS_RECOMMENDATIONS: dict[int, str] = {
    1: "Consider restorative treatment for affected tooth. Schedule follow-up for monitoring.",
    2: "Refer to oral surgeon for evaluation. Monitor for potential complications.",
    3: "Evaluate for extraction or restoration. Consider endodontic consultation.",
    4: "Initiate appropriate antibiotic therapy. Consider endodontic treatment.",
    5: "Evaluate for immediate restoration or extraction. Monitor for pulp exposure.",
    6: "Refer for periodontal evaluation. Consider scaling and root planing.",
    7: "Clinical correlation recommended. Further imaging may be required.",
}

# Clinical finding descriptions by class and severity tier (high / moderate / low)
CLINICAL_DESCRIPTIONS: dict[int, dict[str, str]] = {
    1: {
        "high": "Significant cavity detected on occlusal surface with potential pulp involvement",
        "moderate": "Radiolucency consistent with carious lesion requiring restorative evaluation",
        "low": "Early demineralization or incipient caries suggested; monitoring recommended",
    },
    2: {
        "high": "Fully impacted tooth with clear impaction angle and adjacent structure involvement",
        "moderate": "Partially impacted tooth identified; positional assessment recommended",
        "low": "Possible impaction or ectopic positioning; confirm with clinical examination",
    },
    3: {
        "high": "Severely broken down crown or root with significant structural compromise",
        "moderate": "Crown or root breakdown evident; restoration or extraction planning needed",
        "low": "Minor crown or root defect suggested; monitor for progression",
    },
    4: {
        "high": "Radiographic signs suggest active infection with possible periapical involvement",
        "moderate": "Findings consistent with localized dental infection requiring treatment",
        "low": "Subtle radiographic changes suggestive of infection; clinical correlation advised",
    },
    5: {
        "high": "Fracture line extending through tooth structure with risk of pulp exposure",
        "moderate": "Tooth fracture identified; evaluate stability and restorative options",
        "low": "Possible minor fracture or crack; monitor and correlate clinically",
    },
    6: {
        "high": "Significant alveolar bone loss indicating advanced periodontal involvement",
        "moderate": "Moderate horizontal or vertical bone loss detected in affected region",
        "low": "Early periodontal bone changes noted; periodontal screening recommended",
    },
    7: {
        "high": "Notable radiographic abnormality requiring prompt specialist evaluation",
        "moderate": "Unusual finding detected; further diagnostic workup recommended",
        "low": "Minor radiographic irregularity; follow-up imaging may be helpful",
    },
}


def _severity_tier(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.65:
        return "moderate"
    return "low"


def get_clinical_description(class_id: int, confidence: float) -> str:
    tier = _severity_tier(confidence)
    class_desc = CLINICAL_DESCRIPTIONS.get(class_id, {})
    return class_desc.get(tier) or class_desc.get("moderate") or "Radiographic abnormality detected."


def get_class_recommendation(class_id: int) -> str:
    return CLASS_RECOMMENDATIONS.get(
        class_id,
        "Clinical correlation recommended. Further imaging may be required.",
    )


def _extract_tooth_number(tooth_label: str) -> str | None:
    if "#" not in tooth_label:
        return None
    return tooth_label.split("#")[-1].rstrip(")")


def build_clinical_recommendations(detections: list[dict[str, Any]]) -> list[str]:
    """Aggregate report-level recommendations from all detections."""
    if not detections:
        return [
            "No anomalies detected above threshold — routine follow-up as clinically indicated.",
            "Patient education on oral hygiene practices recommended.",
        ]

    recommendations: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        if text not in seen:
            seen.add(text)
            recommendations.append(text)

    if any(d.get("severity") == "High Severity" for d in detections):
        add("Schedule follow-up examination for high-severity findings")

    if any(d.get("severity") == "Moderate Severity" for d in detections):
        add("Plan clinical evaluation for moderate-severity findings within standard follow-up interval")

    for det in detections:
        tooth_num = _extract_tooth_number(det.get("tooth", ""))
        class_name = det.get("class", "finding")
        if tooth_num:
            add(f"Consider restorative or specialist evaluation for Tooth #{tooth_num} ({class_name.lower()} detected)")
        class_rec = det.get("recommendation") or get_class_recommendation(det.get("class_id", 0))
        add(class_rec)

    add("Patient education on oral hygiene practices recommended")
    return recommendations


def enrich_detection(det: dict[str, Any], img_w: int, img_h: int) -> dict[str, Any]:
    """Add severity, tooth, description, and per-finding recommendation."""
    confidence = float(det.get("confidence", 0))
    class_id = int(det.get("class_id", 0))
    bbox = det.get("bbox", [0, 0, 0, 0])

    enriched = dict(det)
    enriched["severity"] = get_severity(confidence)
    enriched["tooth"] = estimate_tooth(bbox, img_w, img_h)
    enriched["description"] = get_clinical_description(class_id, confidence)
    enriched["recommendation"] = get_class_recommendation(class_id)
    return enriched


def enrich_detections(
    detections: list[dict[str, Any]], img_w: int, img_h: int
) -> dict[str, Any]:
    """Build full analysis payload for API response."""
    enriched = [enrich_detection(d, img_w, img_h) for d in detections]
    return {
        "detections": enriched,
        "clinical_recommendations": build_clinical_recommendations(enriched),
        "footer": WEBSITE_FOOTER,
    }


def ensure_detections_enriched(
    detections: list[dict[str, Any]], img_w: int, img_h: int
) -> list[dict[str, Any]]:
    """Backfill enrichment for older saved analyses missing new fields."""
    if not detections:
        return detections
    if "severity" in detections[0] and "tooth" in detections[0]:
        return detections
    return [enrich_detection(d, img_w, img_h) for d in detections]
