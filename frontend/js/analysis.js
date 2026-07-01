/**
 * DENTRAT — Clinical analysis report UI helpers
 */
const AnalysisUI = {
  severityKey(severity) {
    if (!severity) return "low";
    const s = severity.toLowerCase();
    if (s.includes("high")) return "high";
    if (s.includes("moderate")) return "moderate";
    return "low";
  },

  severityIcon(severity) {
    const key = this.severityKey(severity);
    if (key === "high") return "fa-circle-exclamation";
    if (key === "moderate") return "fa-triangle-exclamation";
    return "fa-circle-info";
  },

  renderFindingCard(det, escFn) {
    const esc = escFn || ((s) => s ?? "");
    const severity = det.severity || "Low Severity";
    const sevClass = this.severityKey(severity);
    const conf = (det.confidence * 100).toFixed(1);
    const borderColor = det.color || CLASS_COLORS[det.class_id] || "#4a6cf7";

    return `
      <li class="clinical-finding-card severity-${sevClass}" style="--finding-accent:${borderColor}">
        <div class="finding-header">
          <span class="finding-class-badge">${esc(det.class)}</span>
          <span class="severity-badge severity-${sevClass}">
            <i class="fa-solid ${this.severityIcon(severity)}"></i> ${esc(severity)}
          </span>
        </div>
        <h4 class="finding-tooth">${esc(det.tooth || det.location || "Approximate location")}</h4>
        <p class="finding-description">${esc(det.description || "Radiographic abnormality detected in the imaged region.")}</p>
        <div class="finding-meta">
          <span class="finding-confidence"><i class="fa-solid fa-chart-simple"></i> Confidence: ${conf}%</span>
        </div>
      </li>`;
  },

  renderFindingsSection(detections, escFn) {
    const esc = escFn || ((s) => s ?? "");
    if (!detections || detections.length === 0) {
      return `
        <section class="clinical-section">
          <h3 class="clinical-section-title"><i class="fa-solid fa-stethoscope"></i> Detected Findings</h3>
          <div class="clinical-empty">No anomalies detected above the confidence threshold.</div>
        </section>`;
    }

    const cards = detections.map((d) => this.renderFindingCard(d, esc)).join("");
    return `
      <section class="clinical-section">
        <h3 class="clinical-section-title"><i class="fa-solid fa-stethoscope"></i> Detected Findings</h3>
        <ul class="clinical-findings-list">${cards}</ul>
      </section>`;
  },

  renderRecommendationsSection(recommendations, escFn) {
    const esc = escFn || ((s) => s ?? "");
    const items = (recommendations || []).map(
      (r) => `<li><i class="fa-solid fa-check"></i><span>${esc(r)}</span></li>`
    ).join("");

    return `
      <section class="clinical-section recommendations-section">
        <h3 class="clinical-section-title"><i class="fa-solid fa-clipboard-list"></i> Clinical Recommendations</h3>
        <ul class="recommendations-list">${items || "<li>No recommendations available.</li>"}</ul>
      </section>`;
  },

  renderFooterNote(footer, escFn) {
    const esc = escFn || ((s) => s ?? "");
    const text = footer || "*This AI-generated report should be reviewed by a licensed dental professional before making clinical decisions.";
    return `<p class="analysis-disclaimer"><em>${esc(text)}</em></p>`;
  },

  renderReport(analysis, escFn) {
    return `
      ${this.renderFindingsSection(analysis.detections, escFn)}
      ${this.renderRecommendationsSection(analysis.clinical_recommendations, escFn)}
      ${this.renderFooterNote(analysis.footer, escFn)}`;
  },
};
