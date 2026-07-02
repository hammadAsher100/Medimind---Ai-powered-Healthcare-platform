const API = {
  baseURL: "/api",
  aiURL: window.MEDIMIND_AI_URL || (window.location.port === "8000" ? "http://localhost:8001" : "/ai"),
  token() { return localStorage.getItem("access_token"); },
  refreshToken() { return localStorage.getItem("refresh_token"); },
  headers(json = true) {
    const headers = {};
    if (json) headers["Content-Type"] = "application/json";
    if (this.token()) headers.Authorization = `Bearer ${this.token()}`;
    return headers;
  },
  async request(endpoint, options = {}, useAI = false) {
    const base = useAI ? this.aiURL : this.baseURL;
    const response = await fetch(`${base}${endpoint}`, options);
    const text = await response.text();
    let payload = {};
    if (text) {
      try { payload = JSON.parse(text); } catch { payload = { detail: text }; }
    }
    if (!response.ok) {
      throw new Error(payload.detail || "The request could not be completed.");
    }
    return payload;
  },
  get(endpoint, useAI = false) {
    return this.request(endpoint, { headers: this.headers(false) }, useAI);
  },
  post(endpoint, data, useAI = false) {
    return this.request(endpoint, {
      method: "POST",
      headers: this.headers(true),
      body: JSON.stringify(data)
    }, useAI);
  },
  upload(endpoint, formData) {
    return this.request(endpoint, {
      method: "POST",
      headers: this.headers(false),
      body: formData
    }, false);
  },
  setTokens(access, refresh) {
    localStorage.setItem("access_token", access);
    localStorage.setItem("refresh_token", refresh);
  },
  clearTokens() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
  },
  isAuthenticated() { return !!this.token(); },
  redirectIfNotAuth() {
    if (!this.isAuthenticated()) window.location = "/login/";
  }
};

function showLoading(message = "Processing your request...") {
  const overlay = document.querySelector("[data-loading-overlay]");
  if (!overlay) return;
  overlay.querySelector("[data-loading-message]").textContent = message;
  overlay.classList.add("active");
}

function hideLoading() {
  const overlay = document.querySelector("[data-loading-overlay]");
  if (overlay) overlay.classList.remove("active");
}
