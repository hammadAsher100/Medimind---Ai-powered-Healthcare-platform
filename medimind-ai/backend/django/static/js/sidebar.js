document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.querySelector(".sidebar");
  const toggle = document.querySelector("[data-sidebar-toggle]");
  if (toggle && sidebar) {
    toggle.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  const current = window.location.pathname;
  document.querySelectorAll(".nav-link").forEach((link) => {
    const href = link.getAttribute("href");
    if (href && href !== "/" && current.startsWith(href)) link.classList.add("active");
    if (href === "/" && (current === "/" || current === "/dashboard/")) link.classList.add("active");
  });

  const signOut = document.querySelector("[data-sign-out]");
  if (signOut) {
    signOut.addEventListener("click", () => API.clearTokens());
  }
});
