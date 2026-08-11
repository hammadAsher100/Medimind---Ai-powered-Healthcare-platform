(() => {
  "use strict";

  const statusText = document.getElementById("statusText");
  const hintText = document.getElementById("hintText");
  const retryButton = document.getElementById("retryButton");
  const progress = document.querySelector("[role='progressbar']");
  const progressBar = document.getElementById("progressBar");
  let startedAt = Date.now();
  const maximumWaitMs = 15 * 60 * 1000;
  let timer = null;
  let stopped = false;

  const progressForStage = {
    "Starting secure server": 28,
    "Preparing a secure restart": 20,
    "Loading MediMind services": 50,
    "Loading AI models": 72,
    "Almost ready": 90,
    "Ready to start": 10,
  };

  function update(message, value) {
    statusText.textContent = message;
    const bounded = Math.max(5, Math.min(96, value || progressForStage[message] || 12));
    progressBar.style.width = `${bounded}%`;
    progress.setAttribute("aria-valuenow", String(bounded));
  }

  function fail(message) {
    stopped = true;
    window.clearTimeout(timer);
    update("Startup paused", 12);
    hintText.textContent = message || "We could not start MediMind right now. Please wait a moment and try again.";
    retryButton.hidden = false;
  }

  async function request(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(path, {
        ...options,
        signal: controller.signal,
        cache: "no-store",
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      });
      const contentType = response.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) throw new Error("Unexpected startup response");
      const payload = await response.json();
      if (!response.ok && payload.status !== "starting" && payload.status !== "error") {
        throw new Error(payload.message || "Startup service unavailable");
      }
      return payload;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function handle(payload) {
    if (payload.status === "ready" && payload.url) {
      stopped = true;
      update("MediMind is ready", 100);
      hintText.textContent = "Opening your secure healthcare workspace…";
      window.setTimeout(() => window.location.assign(payload.url), 500);
      return;
    }
    if (payload.status === "error") {
      fail(payload.message);
      return;
    }
    update(payload.stage || "Preparing MediMind services");
    const elapsed = Date.now() - startedAt;
    if (elapsed >= maximumWaitMs) {
      fail("MediMind is taking longer than expected to start. Please try again.");
      return;
    }
    const interval = Math.max(5000, Math.min(12000, Number(payload.retry_after || 7) * 1000));
    timer = window.setTimeout(poll, interval);
  }

  async function poll() {
    if (stopped) return;
    try {
      handle(await request("/control/status"));
    } catch (error) {
      console.warn("MediMind startup status check failed", error);
      update("Reconnecting to the startup service", 18);
      timer = window.setTimeout(poll, 10000);
    }
  }

  async function wake() {
    stopped = false;
    startedAt = Date.now();
    retryButton.hidden = true;
    hintText.textContent = "You can keep this page open. It will redirect automatically when MediMind is ready.";
    update("Contacting the secure startup service", 12);
    try {
      handle(await request("/control/wake", { method: "POST", body: "{}" }));
    } catch (error) {
      console.error("MediMind wake request failed", error);
      fail("The startup service could not be reached. Check your connection and try again.");
    }
  }

  retryButton.addEventListener("click", wake);
  wake();
})();
