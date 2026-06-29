/**
 * DENTRAT — Main SPA Application
 */

const Router = {
  routes: {},
  navigate(path) { history.pushState({}, "", path); this.render(); },
  register(path, handler) { this.routes[path] = handler; },
  async render() {
    await Auth.init();
    const path = window.location.pathname;

    const publicPaths = ["/login", "/signup"];
    if (!Auth.isLoggedIn() && !publicPaths.includes(path)) {
      history.replaceState({}, "", "/login");
      return this.routes["/login"]?.();
    }
    if (Auth.isLoggedIn() && (path === "/login" || path === "/signup")) {
      history.replaceState({}, "", "/dashboard");
      return this.routes["/dashboard"]?.();
    }

    // Detail route: /saved/123
    const savedMatch = path.match(/^\/saved\/(\d+)$/);
    if (savedMatch) return renderSavedDetail(parseInt(savedMatch[1]));

    const handler = this.routes[path] || this.routes["/dashboard"];
    handler?.();
  },
};

const State = {
  pendingAnalysis: null,  // holds analyze response before save
  uploadedFile: null,
  previewUrl: null,
  helpSection: "getting-started",
};

function showToast(msg, type = "error") {
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.textContent = msg;
  document.getElementById("toast-container").appendChild(t);
  setTimeout(() => t.remove(), 4500);
}

function esc(str) {
  const d = document.createElement("div");
  d.textContent = str ?? "";
  return d.innerHTML;
}

function formatDate(iso) {
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

// ─── Shell ───
function renderShell(activeNav, content) {
  return `
    <div class="app-shell">
      <aside class="sidebar" id="sidebar">
        <div class="sidebar-brand">
          <div class="logo-icon"><i class="fa-solid fa-heart-pulse"></i></div>
          <div><h1>DENTRAT</h1><p>AI Dental Analysis</p></div>
        </div>
        <nav class="sidebar-nav">
          <button class="nav-item ${activeNav === "dashboard" ? "active" : ""}" data-nav="/dashboard">
            <i class="fa-solid fa-microscope"></i> Analysis
          </button>
          <button class="nav-item ${activeNav === "results" ? "active" : ""}" data-nav="/results">
            <i class="fa-solid fa-chart-line"></i> Current Results
          </button>
          <button class="nav-item ${activeNav === "saved" ? "active" : ""}" data-nav="/saved">
            <i class="fa-solid fa-folder-open"></i> Saved Analyses
          </button>
          <button class="nav-item ${activeNav === "help" ? "active" : ""}" data-nav="/help">
            <i class="fa-solid fa-circle-question"></i> Help
          </button>
        </nav>
        <div class="sidebar-footer">© 2026 DENTRAT<br>HIPAA Compliant</div>
      </aside>
      <div class="main-area">
        <header class="topbar">
          <div style="display:flex;align-items:center;gap:1rem">
            <button class="menu-toggle" id="menu-toggle"><i class="fa-solid fa-bars"></i></button>
            <div class="topbar-greeting">
              <h2>Hello, ${esc(Auth.fullName())}</h2>
              <p>Dental Radiography Analysis Dashboard</p>
            </div>
          </div>
          <div class="topbar-actions">
            <span id="health-status" style="font-size:0.8rem;color:var(--gray-600)">
              <span class="health-dot bad"></span> Checking...
            </span>
            <button class="btn btn-secondary btn-sm" id="logout-btn">
              <i class="fa-solid fa-right-from-bracket"></i> Logout
            </button>
          </div>
        </header>
        <main class="page-content">${content}</main>
      </div>
    </div>`;
}

function bindShellEvents() {
  document.querySelectorAll("[data-nav]").forEach((btn) => {
    btn.addEventListener("click", () => Router.navigate(btn.dataset.nav));
  });
  document.getElementById("logout-btn")?.addEventListener("click", async () => {
    await Auth.logout();
    Router.navigate("/login");
  });
  document.getElementById("menu-toggle")?.addEventListener("click", () => {
    document.getElementById("sidebar")?.classList.toggle("open");
  });
  checkHealthStatus();
}

async function checkHealthStatus() {
  const el = document.getElementById("health-status");
  if (!el) return;
  try {
    const h = await API.health();
    if (h.model_loaded) {
      el.innerHTML = '<span class="health-dot ok"></span> AI Model Online';
    } else {
      el.innerHTML = '<span class="health-dot bad"></span> Model Offline';
    }
  } catch {
    el.innerHTML = '<span class="health-dot bad"></span> Server Error';
  }
}

// ─── Sign Up ───
function renderSignup() {
  document.title = "Sign Up — DENTRAT";
  document.getElementById("app").innerHTML = `
    <div class="auth-page">
      <div class="auth-card">
        <div class="auth-brand">
          <div class="auth-logo"><i class="fa-solid fa-heart-pulse"></i></div>
          <h1>DENTRAT</h1>
          <p>AI Dental Radiograph Analyzer</p>
        </div>
        <h2>Create Account</h2>
        <p class="sub">Register to access the analysis dashboard</p>
        <div id="signup-error" class="auth-error hidden"></div>
        <form id="signup-form">
          <div class="form-row"><label>Full Name *</label><input type="text" id="su-name" required placeholder="Dr. John Smith" /></div>
          <div class="form-row"><label>Organization <span class="optional">(optional)</span></label><input type="text" id="su-org" placeholder="Dental Clinic Name" /></div>
          <div class="form-row"><label>Contact <span class="optional">(optional)</span></label><input type="text" id="su-contact" placeholder="+1 555 000 0000" /></div>
          <div class="form-row"><label>Email *</label><input type="email" id="su-email" required placeholder="you@clinic.com" /></div>
          <div class="form-row"><label>Username *</label><input type="text" id="su-user" required placeholder="Unique username" /></div>
          <div class="form-row"><label>Password *</label><input type="password" id="su-pass" required placeholder="Min. 6 characters" /></div>
          <div class="form-row"><label>Confirm Password *</label><input type="password" id="su-confirm" required placeholder="Repeat password" /></div>
          <button type="submit" class="btn btn-primary" id="su-btn"><i class="fa-solid fa-user-plus"></i> Create Account</button>
        </form>
        <p class="auth-switch">Already have an account? <a href="/login" data-link>Sign In</a></p>
      </div>
    </div>`;

  document.querySelector("[data-link]")?.addEventListener("click", (e) => {
    e.preventDefault(); Router.navigate("/login");
  });

  document.getElementById("signup-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = document.getElementById("signup-error");
    const btn = document.getElementById("su-btn");
    errEl.classList.add("hidden");
    btn.disabled = true;

    try {
      await API.signup({
        full_name: document.getElementById("su-name").value.trim(),
        organization: document.getElementById("su-org").value.trim(),
        contact: document.getElementById("su-contact").value.trim(),
        email: document.getElementById("su-email").value.trim(),
        username: document.getElementById("su-user").value.trim(),
        password: document.getElementById("su-pass").value,
        confirm_password: document.getElementById("su-confirm").value,
      });
      showToast("Account created! Please sign in.", "success");
      Router.navigate("/login");
    } catch (err) {
      errEl.textContent = err.message;
      errEl.classList.remove("hidden");
      btn.disabled = false;
    }
  });
}

// ─── Login ───
function renderLogin() {
  document.title = "Sign In — DENTRAT";
  document.getElementById("app").innerHTML = `
    <div class="auth-page">
      <div class="auth-card">
        <div class="auth-brand">
          <div class="auth-logo"><i class="fa-solid fa-heart-pulse"></i></div>
          <h1>DENTRAT</h1>
          <p>AI Dental Radiograph Analyzer</p>
        </div>
        <h2>Sign In</h2>
        <p class="sub">Access your dental analysis dashboard</p>
        <div id="login-error" class="auth-error hidden"></div>
        <form id="login-form">
          <div class="form-row">
            <label>Username or Email</label>
            <div class="input-icon-wrap">
              <i class="fa-solid fa-user"></i>
              <input type="text" id="li-user" required placeholder="Enter username or email" />
            </div>
          </div>
          <div class="form-row">
            <label>Password</label>
            <div class="input-icon-wrap">
              <i class="fa-solid fa-lock"></i>
              <input type="password" id="li-pass" required placeholder="Enter password" />
            </div>
          </div>
          <button type="submit" class="btn btn-primary" id="li-btn"><i class="fa-solid fa-right-to-bracket"></i> Sign In</button>
        </form>
        <p class="auth-switch">Don't have an account? <a href="/signup" data-link>Create Account</a></p>
      </div>
    </div>`;

  document.querySelector("[data-link]")?.addEventListener("click", (e) => {
    e.preventDefault(); Router.navigate("/signup");
  });

  document.getElementById("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = document.getElementById("login-error");
    const btn = document.getElementById("li-btn");
    errEl.classList.add("hidden");
    btn.disabled = true;
    try {
      await Auth.login(
        document.getElementById("li-user").value.trim(),
        document.getElementById("li-pass").value
      );
      Router.navigate("/dashboard");
    } catch (err) {
      errEl.textContent = err.message;
      errEl.classList.remove("hidden");
      btn.disabled = false;
    }
  });
}

// ─── Dashboard ───
function renderDashboard() {
  document.title = "Dashboard — DENTRAT";

  const conditionsHtml = CONDITIONS.map((c) => `
    <div class="condition-item">
      <div class="condition-dot" style="background:${c.color}"><i class="fa-solid ${c.icon}"></i></div>
      <div><span>${c.name}</span></div>
    </div>`).join("");

  const uploadContent = State.previewUrl
    ? `<img src="${State.previewUrl}" class="upload-preview" alt="X-ray" />
       <p style="font-size:0.85rem;color:var(--navy-accent);font-weight:600">${esc(State.uploadedFile?.name)}</p>
       <p style="font-size:0.8rem;color:var(--gray-600)">Click to change</p>`
    : `<div class="upload-icon-circle"><i class="fa-solid fa-cloud-arrow-up"></i></div>
       <h3 style="font-size:0.95rem;font-weight:600;margin-bottom:0.35rem">Drop OPG X-Ray here or click to browse</h3>
       <p style="font-size:0.8rem;color:var(--gray-600)">JPEG, PNG — Max 25 MB</p>`;

  document.getElementById("app").innerHTML = renderShell("dashboard", `
    <div class="hipaa-strip">
      <i class="fa-solid fa-shield-halved"></i>
      <span>All images are processed securely. Analysis complies with HIPAA regulations. No data shared without consent.</span>
    </div>
    <div class="dashboard-grid">
      <div class="panel">
        <div class="panel-header">
          <h3><i class="fa-solid fa-list-check"></i> Dental conditions we can detect</h3>
          <p>Our AI automatically scans for all conditions below</p>
        </div>
        <div class="condition-list">${conditionsHtml}</div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <h3><i class="fa-solid fa-x-ray"></i> Upload OPG X-Ray</h3>
          <p>Upload an image — analysis runs automatically</p>
        </div>
        <div class="upload-zone ${State.previewUrl ? "has-file" : ""}" id="upload-zone">
          <input type="file" id="file-input" accept="image/*" hidden />
          ${uploadContent}
        </div>
      </div>
    </div>
    <div id="analyze-overlay" class="overlay hidden">
      <div class="overlay-card">
        <div class="spinner"></div>
        <h3>Analyzing X-Ray</h3>
        <p>Running AI detection model...</p>
        <div class="progress-bar"><div class="progress-fill" id="prog-fill" style="width:0%"></div></div>
      </div>
    </div>`);

  bindShellEvents();
  bindUploadEvents();
}

function bindUploadEvents() {
  const zone = document.getElementById("upload-zone");
  const input = document.getElementById("file-input");
  if (!zone) return;

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault(); zone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", (e) => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
  });
}

async function handleUpload(file) {
  if (!file.type.startsWith("image/")) return showToast("Please upload a valid image.");
  if (file.size > 25 * 1024 * 1024) return showToast("File too large (max 25 MB).");

  State.uploadedFile = file;
  State.previewUrl = URL.createObjectURL(file);

  const overlay = document.getElementById("analyze-overlay");
  const fill = document.getElementById("prog-fill");
  overlay?.classList.remove("hidden");

  let prog = 5;
  const timer = setInterval(() => {
    prog = Math.min(prog + Math.random() * 12, 85);
    if (fill) fill.style.width = `${prog}%`;
  }, 300);

  try {
    const data = await API.analyze(file, (pct) => { if (fill) fill.style.width = `${Math.max(pct, 10)}%`; });
    clearInterval(timer);
    if (fill) fill.style.width = "100%";

    State.pendingAnalysis = { ...data, previewUrl: State.previewUrl, file };
    await new Promise((r) => setTimeout(r, 400));
    overlay?.classList.add("hidden");
    showToast(`Analysis complete — ${data.detection_count} finding(s)`, "success");
    Router.navigate("/results");
  } catch (err) {
    clearInterval(timer);
    overlay?.classList.add("hidden");
    showToast(err.message);
    renderDashboard();
  }
}

// ─── Results (current analysis + patient form) ───
function renderResults() {
  document.title = "Results — DENTRAT";
  const a = State.pendingAnalysis;

  if (!a) {
    document.getElementById("app").innerHTML = renderShell("results", `
      <div class="empty-state">
        <i class="fa-solid fa-file-medical"></i>
        <h3>No Current Analysis</h3>
        <p>Upload an X-ray from the Analysis page to see results here.</p>
        <button class="btn btn-primary btn-sm" style="margin-top:1rem;width:auto" data-nav="/dashboard">
          <i class="fa-solid fa-upload"></i> Go to Analysis
        </button>
      </div>`);
    bindShellEvents();
    return;
  }

  const findingsHtml = a.detections.length === 0
    ? `<li class="finding-card" style="border-left-color:var(--gray-400)"><h4>No anomalies detected</h4><p style="font-size:0.8rem;color:var(--gray-600)">Above 50% confidence threshold</p></li>`
    : a.detections.map((d) => `
        <li class="finding-card" style="border-left-color:${d.color || CLASS_COLORS[d.class_id]}">
          <h4>${esc(d.class)}</h4>
          <div class="conf">${Math.round(d.confidence * 100)}% confidence</div>
          <div class="loc"><i class="fa-solid fa-location-dot"></i> ${esc(d.location)}</div>
        </li>`).join("");

  document.getElementById("app").innerHTML = renderShell("results", `
    <div class="panel" style="margin-bottom:1.5rem">
      <div class="panel-header">
        <h3><i class="fa-solid fa-chart-line"></i> Analysis Results — ${a.detection_count} Finding(s)</h3>
        <p>${esc(a.filename)} · ${formatDate(a.analysis_date)}</p>
      </div>
      <div class="results-layout">
        <div class="canvas-wrap"><canvas id="result-canvas"></canvas></div>
        <div>
          <ul class="findings-list">${findingsHtml}</ul>
          <div class="patient-form">
            <h4><i class="fa-solid fa-user-injured"></i> Patient Information</h4>
            <div class="form-grid">
              <div class="form-row"><label>Patient Name <span class="optional">(optional)</span></label>
                <input type="text" id="pt-name" placeholder="Patient full name" /></div>
              <div class="form-row"><label>Contact <span class="optional">(optional)</span></label>
                <input type="text" id="pt-contact" placeholder="Phone number" /></div>
              <div class="form-row full"><label>Email <span class="optional">(optional)</span></label>
                <input type="email" id="pt-email" placeholder="patient@email.com" /></div>
              <div class="form-row full"><label>Date & Time</label>
                <div class="auto-date"><i class="fa-solid fa-clock"></i> ${formatDate(a.analysis_date)} (auto-generated)</div></div>
            </div>
            <div class="action-row">
              <button class="btn btn-primary" id="btn-save"><i class="fa-solid fa-floppy-disk"></i> Save Analysis</button>
              <button class="btn btn-outline" id="btn-save-pdf"><i class="fa-solid fa-file-pdf"></i> Save & Generate PDF</button>
            </div>
          </div>
        </div>
      </div>
    </div>`);

  bindShellEvents();
  drawCanvas(a.previewUrl, a.detections, a.image_width, a.image_height);

  document.getElementById("btn-save").addEventListener("click", () => saveCurrentAnalysis(false));
  document.getElementById("btn-save-pdf").addEventListener("click", () => saveCurrentAnalysis(true));
}

async function saveCurrentAnalysis(downloadPdf) {
  const a = State.pendingAnalysis;
  if (!a) return;

  const payload = {
    temp_image_path: a.temp_image_path,
    detections: a.detections,
    analysis_date: a.analysis_date,
    patient_name: document.getElementById("pt-name")?.value.trim(),
    patient_contact: document.getElementById("pt-contact")?.value.trim(),
    patient_email: document.getElementById("pt-email")?.value.trim(),
  };

  try {
    const data = await API.saveAnalysis(payload);
    showToast("Analysis saved successfully!", "success");
    State.pendingAnalysis = null;

    if (downloadPdf && data.analysis?.id) {
      window.open(API.pdfUrl(data.analysis.id), "_blank");
    }
    Router.navigate("/saved");
  } catch (err) {
    showToast(err.message);
  }
}

function drawCanvas(dataUrl, detections, w, h) {
  const canvas = document.getElementById("result-canvas");
  if (!canvas) return;
  const img = new Image();
  img.onload = () => {
    const maxW = 560;
    const scale = Math.min(1, maxW / img.width);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const sx = canvas.width / (w || img.width);
    const sy = canvas.height / (h || img.height);
    detections.forEach((det) => {
      const [x, y, bw, bh] = det.bbox;
      const color = det.color || CLASS_COLORS[det.class_id] || "#f00";
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.strokeRect(x * sx, y * sy, bw * sx, bh * sy);
      const label = `${det.class} ${Math.round(det.confidence * 100)}%`;
      ctx.font = "bold 11px Inter,sans-serif";
      ctx.fillStyle = color;
      ctx.fillRect(x * sx, Math.max(y * sy - 14, 0), ctx.measureText(label).width + 6, 14);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x * sx + 3, Math.max(y * sy - 3, 10));
    });
  };
  img.src = dataUrl;
}

// ─── Saved Analyses ───
async function renderSaved() {
  document.title = "Saved Analyses — DENTRAT";
  document.getElementById("app").innerHTML = renderShell("saved", `
    <div class="panel"><div class="panel-header">
      <h3><i class="fa-solid fa-folder-open"></i> Saved Analyses</h3>
      <p>All saved patient analyses</p>
    </div><div id="saved-list"><div class="spinner" style="margin:2rem auto"></div></div></div>`);
  bindShellEvents();

  try {
    const { analyses } = await API.getSavedAnalyses();
    const el = document.getElementById("saved-list");
    if (!analyses.length) {
      el.innerHTML = `<div class="empty-state"><i class="fa-solid fa-folder-open"></i>
        <h3>No Saved Analyses</h3><p>Complete an analysis and save it to see it here.</p></div>`;
      return;
    }
    el.innerHTML = `<div class="saved-list">${analyses.map((a) => `
      <div class="saved-card" data-id="${a.id}">
        <div>
          <h4>${esc(a.patient_name)}</h4>
          <div class="meta"><i class="fa-solid fa-calendar"></i> ${formatDate(a.analysis_date)}</div>
        </div>
        <span class="badge"><i class="fa-solid fa-bullseye"></i> ${a.detection_count} detections</span>
      </div>`).join("")}</div>`;

    el.querySelectorAll(".saved-card").forEach((card) => {
      card.addEventListener("click", () => Router.navigate(`/saved/${card.dataset.id}`));
    });
  } catch (err) {
    showToast(err.message);
  }
}

async function renderSavedDetail(id) {
  if (!Auth.isLoggedIn()) return Router.navigate("/login");
  document.title = "Analysis Detail — DENTRAT";

  document.getElementById("app").innerHTML = renderShell("saved", `
    <div id="detail-content"><div class="spinner" style="margin:3rem auto"></div></div>`);
  bindShellEvents();

  try {
    const { analysis: a } = await API.getAnalysis(id);
    const findingsHtml = a.detections.map((d) => `
      <li class="finding-card" style="border-left-color:${CLASS_COLORS[d.class_id]}">
        <h4>${esc(d.class)}</h4>
        <div class="conf">${Math.round(d.confidence * 100)}% confidence</div>
        <div class="loc">${esc(d.location || "")}</div>
      </li>`).join("");

    document.getElementById("detail-content").innerHTML = `
      <button class="btn btn-secondary btn-sm" id="back-btn" style="margin-bottom:1rem">
        <i class="fa-solid fa-arrow-left"></i> Back
      </button>
      <div class="panel">
        <div class="panel-header">
          <h3>${esc(a.patient_name || "Unnamed Patient")}</h3>
          <p>${formatDate(a.analysis_date)} · ${a.detection_count} detection(s)</p>
        </div>
        <div class="form-grid" style="margin-bottom:1rem">
          <div><strong style="font-size:0.8rem;color:var(--gray-600)">Contact</strong><p>${esc(a.patient_contact || "—")}</p></div>
          <div><strong style="font-size:0.8rem;color:var(--gray-600)">Email</strong><p>${esc(a.patient_email || "—")}</p></div>
        </div>
        <ul class="findings-list">${findingsHtml || "<li>No detections</li>"}</ul>
        <div style="margin-top:1.25rem">
          <label style="font-size:0.85rem;font-weight:600">Clinical Notes / Comment</label>
          <textarea class="comment-box" id="comment-input" placeholder="Add notes about this analysis...">${esc(a.comment || "")}</textarea>
        </div>
        <div class="detail-actions">
          <button class="btn btn-primary btn-sm" id="btn-pdf"><i class="fa-solid fa-file-pdf"></i> Download PDF</button>
          <button class="btn btn-secondary btn-sm" id="btn-comment"><i class="fa-solid fa-comment"></i> Save Comment</button>
          <button class="btn btn-danger btn-sm" id="btn-delete"><i class="fa-solid fa-trash"></i> Delete</button>
        </div>
      </div>`;

    document.getElementById("back-btn").addEventListener("click", () => Router.navigate("/saved"));
    document.getElementById("btn-pdf").addEventListener("click", () => window.open(API.pdfUrl(id), "_blank"));
    document.getElementById("btn-comment").addEventListener("click", async () => {
      try {
        await API.updateComment(id, document.getElementById("comment-input").value);
        showToast("Comment saved!", "success");
      } catch (err) { showToast(err.message); }
    });
    document.getElementById("btn-delete").addEventListener("click", async () => {
      if (!confirm("Delete this analysis permanently?")) return;
      try {
        await API.deleteAnalysis(id);
        showToast("Analysis deleted.", "success");
        Router.navigate("/saved");
      } catch (err) { showToast(err.message); }
    });
  } catch (err) {
    showToast(err.message);
    Router.navigate("/saved");
  }
}

// ─── Help ───
const HELP = {
  "getting-started": { title: "Getting Started", body: `<p>Welcome to DENTRAT. After signing in:</p><ol><li>Go to <strong>Analysis</strong></li><li>Upload an OPG X-ray (auto-analyzes)</li><li>Review results and enter patient info</li><li>Save analysis and generate PDF report</li></ol>` },
  upload: { title: "Uploading X-Rays", body: `<p>Supported: JPEG, PNG, BMP, TIFF, WEBP. Max 25 MB. No condition selection needed — all 7 anomaly types are scanned automatically.</p>` },
  pdf: { title: "PDF Reports", body: `<p>After analysis, enter patient details and click <strong>Save & Generate PDF</strong>. Reports include DENTRAT branding, patient info, detection table, and annotated X-ray.</p>` },
  saved: { title: "Saved Analyses", body: `<p>All saved analyses appear in the Saved tab. View details, add comments, download PDFs, or delete records.</p>` },
};

function renderHelp() {
  document.title = "Help — DENTRAT";
  const section = HELP[State.helpSection] || HELP["getting-started"];
  const navHtml = Object.entries(HELP).map(([k, v]) =>
    `<button class="help-nav-item ${State.helpSection === k ? "active" : ""}" data-help="${k}">${v.title}</button>`
  ).join("");

  document.getElementById("app").innerHTML = renderShell("help", `
    <div class="help-grid">
      <div class="panel"><div class="help-nav">${navHtml}</div></div>
      <div class="panel help-body"><h3>${section.title}</h3>${section.body}</div>
    </div>`);
  bindShellEvents();
  document.querySelectorAll("[data-help]").forEach((btn) => {
    btn.addEventListener("click", () => { State.helpSection = btn.dataset.help; renderHelp(); });
  });
}

// ─── Init ───
Router.register("/signup", renderSignup);
Router.register("/login", renderLogin);
Router.register("/dashboard", renderDashboard);
Router.register("/", renderDashboard);
Router.register("/results", renderResults);
Router.register("/saved", renderSaved);
Router.register("/help", renderHelp);

window.addEventListener("popstate", () => Router.render());
document.addEventListener("DOMContentLoaded", () => Router.render());
