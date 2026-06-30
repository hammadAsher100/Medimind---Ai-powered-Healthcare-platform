const notify = (() => {
  function region() {
    let el = document.querySelector(".toast-region");
    if (!el) {
      el = document.createElement("div");
      el.className = "toast-region";
      document.body.appendChild(el);
    }
    return el;
  }

  function push(type, message) {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    region().appendChild(toast);
    window.setTimeout(() => toast.remove(), 4000);
  }

  return {
    success: (message) => push("success", message),
    error: (message) => push("error", message),
    warning: (message) => push("warning", message),
    info: (message) => push("info", message)
  };
})();
