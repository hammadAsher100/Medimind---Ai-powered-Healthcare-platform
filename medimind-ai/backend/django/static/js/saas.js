document.addEventListener("DOMContentLoaded", () => {
  const current = window.location.pathname;

  document.querySelectorAll(".nav-link").forEach((link) => {
    const href = link.getAttribute("href");
    if (!href) return;
    const active = href === "/" ? current === "/" || current === "/dashboard/" : current.startsWith(href);
    link.classList.toggle("active", active);
  });

  const navToggle = document.querySelector("[data-nav-toggle]");
  const navMenu = document.querySelector("[data-nav-menu]");
  if (navToggle && navMenu) {
    navToggle.addEventListener("click", () => {
      const open = navMenu.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
  }

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

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        userMenu.classList.remove("open");
        userToggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  const signOut = document.querySelector("[data-sign-out]");
  if (signOut && window.API) {
    signOut.addEventListener("click", () => API.clearTokens());
  }
});
