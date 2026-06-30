const chartPalette = {
  accent: "#1E88E5",
  accentLight: "#42A5F5",
  success: "#10B981",
  warning: "#F59E0B",
  danger: "#EF4444",
  border: "#E2E8F0",
  text: "#334155"
};

function makeLineChart(canvas, labels, values) {
  if (!canvas || !window.Chart) return null;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, 260);
  gradient.addColorStop(0, "rgba(30,136,229,0.28)");
  gradient.addColorStop(1, "rgba(30,136,229,0)");
  return new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: chartPalette.accent,
        backgroundColor: gradient,
        fill: true,
        tension: 0.38,
        pointRadius: 4,
        pointBackgroundColor: chartPalette.accent
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, grid: { color: chartPalette.border } },
        x: { grid: { display: false } }
      }
    }
  });
}

function makeHorizontalBarChart(canvas, labels, values, directions) {
  if (!canvas || !window.Chart) return null;
  const colors = values.map((_, index) => directions[index] === "decreases_risk" ? chartPalette.success : chartPalette.danger);
  return new Chart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 6 }] },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, grid: { color: chartPalette.border } },
        y: { grid: { display: false } }
      }
    }
  });
}
