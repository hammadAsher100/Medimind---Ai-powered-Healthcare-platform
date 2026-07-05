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

function makeBarChart(canvas, labels, values, colors) {
  if (!canvas || !window.Chart) return null;
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors || [chartPalette.accent, chartPalette.success, chartPalette.warning, chartPalette.info],
        borderRadius: 8,
        maxBarThickness: 46
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: chartPalette.border } },
        x: { grid: { display: false } }
      }
    }
  });
}

function makeDoughnutChart(canvas, labels, values, colors) {
  if (!canvas || !window.Chart) return null;
  return new Chart(canvas, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors || [chartPalette.success, chartPalette.warning, chartPalette.danger, chartPalette.accent],
        borderColor: "#FFFFFF",
        borderWidth: 4,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 10, usePointStyle: true }
        }
      }
    }
  });
}

function makeRadarChart(canvas, labels, values) {
  if (!canvas || !window.Chart) return null;
  return new Chart(canvas, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: chartPalette.accent,
        backgroundColor: "rgba(30,136,229,0.16)",
        pointBackgroundColor: chartPalette.accent,
        pointBorderColor: "#FFFFFF",
        pointRadius: 4,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { display: false, stepSize: 20 },
          grid: { color: chartPalette.border },
          angleLines: { color: chartPalette.border },
          pointLabels: { color: chartPalette.text, font: { size: 12, weight: "600" } }
        }
      }
    }
  });
}
