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
    let response;
    try {
      response = await fetch(`${base}${endpoint}`, options);
    } catch (cause) {
      console.error("API network request failed", { endpoint, cause });
      const error = new Error("Unable to contact the server. Check your connection and try again.");
      error.cause = cause;
      throw error;
    }

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

    const contentType = (response.headers.get("Content-Type") || "").toLowerCase();
    const isJSON = contentType.includes("application/json") || contentType.includes("+json");
    const text = await response.text();
    let payload = {};
    if (text && isJSON) {
      try {
        payload = JSON.parse(text);
      } catch (cause) {
        console.error("API returned malformed JSON", {
          endpoint,
          status: response.status,
          contentType,
          cause,
        });
        const error = new Error("The server returned an invalid response. Please try again.");
        error.status = response.status;
        throw error;
      }
    } else if (text && !isJSON) {
      console.error("API returned an unexpected non-JSON response", {
        endpoint,
        status: response.status,
        contentType: contentType || "missing",
      });
    }

    if (!response.ok) {
      const detail = isJSON ? (payload.error || payload.detail) : null;
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
      error.status = response.status;
      error.contentType = contentType;
      throw error;
    }

    if (text && !isJSON) {
      const error = new Error("The server returned an invalid response. Please try again.");
      error.status = response.status;
      error.contentType = contentType;
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
