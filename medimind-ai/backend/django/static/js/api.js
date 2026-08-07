const API = {
  baseURL: "/api",
  aiURL: window.MEDIMIND_AI_URL || (window.location.port === "8000" ? "http://localhost:8001" : "/ai"),
  token() { return localStorage.getItem("access_token") || sessionStorage.getItem("access_token"); },
  refreshToken() { return localStorage.getItem("refresh_token") || sessionStorage.getItem("refresh_token"); },
  headers(json = true) {
    const headers = {};
    if (json) headers["Content-Type"] = "application/json";
    if (this.token()) headers.Authorization = `Bearer ${this.token()}`;
    const csrfMatch = document.cookie.match(/csrftoken=([^;]+)/);
    if (csrfMatch) headers["X-CSRFToken"] = csrfMatch[1];
    return headers;
  },
  async _tryRefreshToken() {
    const refresh = this.refreshToken();
    if (!refresh) return false;
    try {
      const res = await fetch("/api/auth/token/refresh/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh }),
        credentials: "same-origin",
      });
      if (!res.ok) return false;
      const data = await res.json();
      if (data.access) {
        const storage = localStorage.getItem("refresh_token") ? localStorage : sessionStorage;
        storage.setItem("access_token", data.access);
        if (data.refresh) storage.setItem("refresh_token", data.refresh);
        return true;
      }
    } catch { /* ignore */ }
    return false;
  },
  async request(endpoint, options = {}, useAI = false) {
    const base = useAI ? this.aiURL : this.baseURL;
    // Always send cookies so SessionAuthentication works as fallback
    if (!useAI) options.credentials = "same-origin";
    const response = await fetch(`${base}${endpoint}`, options);

    // If 401 and we have a refresh token, try refreshing and retry once
    if (response.status === 401 && !options._retried && !useAI) {
      const refreshed = await this._tryRefreshToken();
      if (refreshed) {
        // Rebuild headers with the new token
        const newHeaders = {};
        if (options.headers) {
          for (const [k, v] of Object.entries(options.headers)) {
            newHeaders[k] = v;
          }
        }
        newHeaders.Authorization = `Bearer ${this.token()}`;
        options.headers = newHeaders;
        options._retried = true;
        return this.request(endpoint, options, useAI);
      }
    }

    const text = await response.text();
    let payload = {};
    if (text) {
      try { payload = JSON.parse(text); } catch { payload = { detail: text }; }
    }
    if (!response.ok) {
      const detail = payload.detail;
      let message = "The request could not be completed.";
      if (typeof detail === "string") {
        message = detail;
      } else if (detail && typeof detail === "object") {
        message = Object.values(detail).flat().join(" ");
      } else if (payload && typeof payload === "object") {
        const fieldErrors = Object.entries(payload)
          .flatMap(([field, errors]) => (Array.isArray(errors) ? errors : [errors]).map((error) => `${field.replaceAll("_", " ")}: ${error}`));
        if (fieldErrors.length) message = fieldErrors.join(" ");
      }
      const error = new Error(message);
      error.payload = payload;
      throw error;
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
  uploadAI(endpoint, formData) {
    return this.request(endpoint, {
      method: "POST",
      headers: this.headers(false),
      body: formData
    }, true);
  },
  setTokens(access, refresh, persistent = true) {
    this.clearTokens();
    const storage = persistent ? localStorage : sessionStorage;
    storage.setItem("access_token", access);
    storage.setItem("refresh_token", refresh);
  },
  clearTokens() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    sessionStorage.removeItem("access_token");
    sessionStorage.removeItem("refresh_token");
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
  overlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("is-loading");
}

function hideLoading() {
  const overlay = document.querySelector("[data-loading-overlay]");
  if (overlay) {
    overlay.classList.remove("active");
    overlay.setAttribute("aria-hidden", "true");
  }
  document.body.classList.remove("is-loading");
}
