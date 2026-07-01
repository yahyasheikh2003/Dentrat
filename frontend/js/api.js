/**
 * DENTRAT API client — all backend communication.
 */
const API = {
  async request(url, options = {}) {
    const res = await fetch(url, {
      credentials: "include",
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || data.detail || "Request failed");
    return data;
  },

  health() {
    return fetch("/health").then((r) => r.json());
  },

  me() {
    return fetch("/me", { credentials: "include" }).then((r) => r.json());
  },

  signup(payload) {
    return this.request("/signup", { method: "POST", body: JSON.stringify(payload) });
  },

  login(username, password) {
    return this.request("/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
  },

  logout() {
    return this.request("/logout", { method: "POST" });
  },

  stats() {
    return fetch("/stats", { credentials: "include" }).then((r) => r.json());
  },

  analyze(file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const fd = new FormData();
      fd.append("image", file);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        try {
          const data = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) resolve(data);
          else reject(new Error(data.error || "Analysis failed"));
        } catch { reject(new Error("Invalid response")); }
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.open("POST", "/analyze");
      xhr.withCredentials = true;
      xhr.send(fd);
    });
  },

  saveAnalysis(payload) {
    return this.request("/save-analysis", { method: "POST", body: JSON.stringify(payload) });
  },

  getSavedAnalyses() {
    return this.request("/saved-analyses");
  },

  getAnalysis(id) {
    return this.request(`/analysis/${id}`);
  },

  deleteAnalysis(id) {
    return this.request(`/analysis/${id}`, { method: "DELETE" });
  },

  updateComment(id, comment) {
    return this.request(`/analysis/${id}/comment`, {
      method: "PUT",
      body: JSON.stringify({ comment }),
    });
  },

  pdfUrl(id) {
    return `/analysis/${id}/pdf`;
  },

  submitContact(payload) {
    return this.request("/contact", { method: "POST", body: JSON.stringify(payload) });
  },
};

const CLASS_COLORS = {
  1: "#FF4444", 2: "#FF8800", 3: "#FFCC00", 4: "#44AA44",
  5: "#4488FF", 6: "#AA44FF", 7: "#3b82f6",
};

const CONDITIONS = [
  { id: 1, name: "Caries & Cavities", icon: "fa-circle", color: "#ef4444" },
  { id: 2, name: "Impacted Teeth", icon: "fa-tooth", color: "#f97316" },
  { id: 6, name: "Periodontal Bone Loss", icon: "fa-bone", color: "#a855f7" },
  { id: 3, name: "Broken Crown/Root", icon: "fa-shield-halved", color: "#eab308" },
  { id: 4, name: "Infection", icon: "fa-virus", color: "#ec4899" },
  { id: 5, name: "Fractured Teeth", icon: "fa-bolt", color: "#22c55e" },
  { id: 7, name: "Other Abnormalities", icon: "fa-magnifying-glass", color: "#3b82f6" },
];
