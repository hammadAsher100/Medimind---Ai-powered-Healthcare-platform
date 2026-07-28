document.addEventListener("DOMContentLoaded", () => {
  const current = window.location.pathname;

  // ── Active nav link ──────────────────────────────────
  document.querySelectorAll(".nav-link").forEach((link) => {
    const href = link.getAttribute("href");
    if (!href) return;
    const active = href === "/" ? current === "/" || current === "/dashboard/" : current.startsWith(href);
    link.classList.toggle("active", active);
  });

  // Also highlight dropdown items whose parent dropdown contains the active link
  document.querySelectorAll(".nav-dropdown").forEach((dropdown) => {
    const activeItem = dropdown.querySelector(".dropdown-item.active");
    if (activeItem) {
      const trigger = dropdown.querySelector(".nav-dropdown-trigger");
      if (trigger) trigger.classList.add("active");
    }
  });

  document.querySelectorAll(".dropdown-item").forEach((item) => {
    const href = item.getAttribute("href");
    if (!href) return;
    const active = href === "/" ? current === "/" || current === "/dashboard/" : current.startsWith(href);
    item.classList.toggle("active", active);
  });

  // ── Mobile nav toggle ────────────────────────────────
  const navToggle = document.querySelector("[data-nav-toggle]");
  const navMenu = document.querySelector("[data-nav-menu]");
  if (navToggle && navMenu) {
    navToggle.addEventListener("click", () => {
      const open = navMenu.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
  }

  // ── Nav dropdowns ────────────────────────────────────
  // Single delegated handler — avoids event-propagation races
  document.addEventListener("click", (e) => {
    const trigger = e.target.closest("[data-dropdown-toggle]");
    const dropdown = e.target.closest("[data-nav-dropdown]");

    if (trigger && dropdown) {
      // Toggle this dropdown
      const panel = dropdown.querySelector("[data-dropdown-panel]");
      const isOpen = dropdown.classList.contains("open");

      // Close every other open dropdown first
      document.querySelectorAll("[data-nav-dropdown].open").forEach((other) => {
        if (other !== dropdown) {
          other.classList.remove("open");
          const ot = other.querySelector("[data-dropdown-toggle]");
          if (ot) ot.setAttribute("aria-expanded", "false");
          const op = other.querySelector("[data-dropdown-panel]");
          if (op) op.classList.remove("open");
        }
      });

      if (isOpen) {
        dropdown.classList.remove("open");
        if (panel) panel.classList.remove("open");
        trigger.setAttribute("aria-expanded", "false");
      } else {
        dropdown.classList.add("open");
        if (panel) panel.classList.add("open");
        trigger.setAttribute("aria-expanded", "true");
      }
      return; // handled — don't fall through to close logic
    }

    // Click outside any dropdown → close all
    if (!dropdown) {
      document.querySelectorAll("[data-nav-dropdown].open").forEach((dd) => {
        dd.classList.remove("open");
        const t = dd.querySelector("[data-dropdown-toggle]");
        if (t) t.setAttribute("aria-expanded", "false");
        const p = dd.querySelector("[data-dropdown-panel]");
        if (p) p.classList.remove("open");
      });
    }
  });

  // ── User menu ────────────────────────────────────────
  const userToggle = document.querySelector("[data-user-menu-toggle]");
  const userMenu = document.querySelector("[data-user-menu]");
  if (userToggle && userMenu) {
    userToggle.addEventListener("click", (event) => {
      event.stopPropagation();
      const open = userMenu.classList.toggle("open");
      userToggle.setAttribute("aria-expanded", String(open));
    });

    document.addEventListener("click", (event) => {
      if (!userMenu.contains(event.target) && !userToggle.contains(event.target)) {
        userMenu.classList.remove("open");
        userToggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  // ── Escape key closes all open panels ────────────────
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      // Close dropdowns
      document.querySelectorAll("[data-nav-dropdown].open").forEach((dd) => {
        dd.classList.remove("open");
        const trigger = dd.querySelector("[data-dropdown-toggle]");
        if (trigger) trigger.setAttribute("aria-expanded", "false");
        const panel = dd.querySelector("[data-dropdown-panel]");
        if (panel) panel.classList.remove("open");
      });
      // Close mobile nav
      if (navMenu && navMenu.classList.contains("open")) {
        navMenu.classList.remove("open");
        if (navToggle) navToggle.setAttribute("aria-expanded", "false");
      }
      // Close user menu
      if (userMenu && userMenu.classList.contains("open")) {
        userMenu.classList.remove("open");
        if (userToggle) userToggle.setAttribute("aria-expanded", "false");
      }
    }
  });

  const signOut = document.querySelector("[data-sign-out]");
  if (signOut && window.API) {
    signOut.addEventListener("click", () => API.clearTokens());
  }
});
