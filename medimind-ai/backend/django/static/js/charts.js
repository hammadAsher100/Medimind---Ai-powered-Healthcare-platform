/*
 * MediMind AI — Chart System
 * Clinical green palette (#194D3A). No gradient fills.
 * Canvas fallbacks use flat fills. Viridis/inferno for heatmaps.
 */

const chartPalette = {
  accent: "#194D3A",
  accentLight: "#2D7A5E",
  success: "#194D3A",
  warning: "#B45309",
  danger: "#9B2C2C",
  info: "#1E40AF",
  border: "#E5E4E0",
  text: "#151714",
  muted: "#90948B",
  surface: "#F4F3EF",
  canvas: "#F4F3EF",
  white: "#FFFFFF"
};

/* Viridis-inspired palette for heatmaps */
const viridisPalette = [
  "#440154", "#482878", "#3E4989", "#31688E", "#26828E",
  "#1F9E89", "#35B779", "#6ECE58", "#B5DE2B", "#FDE725"
];

/* Inferno-inspired palette for heatmaps */
const infernoPalette = [
  "#000004", "#160B39", "#420A68", "#6A176E", "#932667",
  "#BC3754", "#DD513A", "#F37819", "#FCA50A", "#F6D746"
];

function prepareCanvas(canvas) {
  if (!canvas) return null;
  const parent = canvas.parentElement;
  const rect = parent ? parent.getBoundingClientRect() : canvas.getBoundingClientRect();
  const width = Math.max(280, Math.floor(rect.width || 640));
  const height = Math.max(220, Math.floor(rect.height || 300));
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  return { ctx, width, height };
}

function numericValues(values) {
  return (values || []).map(value => Number(value) || 0);
}

function drawFallbackEmpty(canvas, message = "No chart data yet") {
  const prepared = prepareCanvas(canvas);
  if (!prepared) return null;
  const { ctx, width, height } = prepared;
  ctx.fillStyle = chartPalette.canvas;
  ctx.strokeStyle = chartPalette.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(18, 18, width - 36, height - 36, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = chartPalette.muted;
  ctx.textAlign = "center";
  ctx.font = "500 13px Inter, system-ui, sans-serif";
  ctx.fillText(message, width / 2, height / 2);
  return { fallback: true };
}

function drawLineFallback(canvas, labels, values) {
  const data = numericValues(values);
  if (!data.length) return drawFallbackEmpty(canvas);
  const prepared = prepareCanvas(canvas);
  if (!prepared) return null;
  const { ctx, width, height } = prepared;
  const pad = { top: 20, right: 26, bottom: 34, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const max = Math.max(100, ...data);
  const min = 0;

  ctx.strokeStyle = chartPalette.border;
  ctx.lineWidth = 1;
  ctx.fillStyle = chartPalette.muted;
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (plotH / 4) * i;
    const value = Math.round(max - ((max - min) / 4) * i);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(String(value), pad.left - 10, y + 4);
  }

  const points = data.map((value, index) => {
    const x = pad.left + (data.length === 1 ? plotW / 2 : (plotW / (data.length - 1)) * index);
    const y = pad.top + plotH - ((value - min) / (max - min || 1)) * plotH;
    return { x, y };
  });

  /* Flat area fill — no gradient */
  ctx.beginPath();
  points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.lineTo(points[points.length - 1].x, height - pad.bottom);
  ctx.lineTo(points[0].x, height - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = "rgba(25, 77, 58, 0.08)";
  ctx.fill();

  ctx.beginPath();
  points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.strokeStyle = chartPalette.accent;
  ctx.lineWidth = 2;
  ctx.stroke();

  points.forEach(point => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = chartPalette.white;
    ctx.fill();
    ctx.strokeStyle = chartPalette.accent;
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  ctx.textAlign = "center";
  ctx.fillStyle = chartPalette.muted;
  const visibleLabels = labels || [];
  points.forEach((point, index) => {
    if (index === 0 || index === points.length - 1 || points.length <= 6) {
      ctx.fillText(visibleLabels[index] || "", point.x, height - 12);
    }
  });
  return { fallback: true };
}

function drawBarFallback(canvas, labels, values, colors, horizontal = false) {
  const data = numericValues(values);
  if (!data.length) return drawFallbackEmpty(canvas);
  const prepared = prepareCanvas(canvas);
  if (!prepared) return null;
  const { ctx, width, height } = prepared;
  const palette = colors || [chartPalette.accent, chartPalette.success, chartPalette.warning, chartPalette.info];
  const pad = { top: 20, right: 28, bottom: horizontal ? 18 : 42, left: horizontal ? 120 : 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const max = Math.max(10, ...data);

  ctx.strokeStyle = chartPalette.border;
  ctx.fillStyle = chartPalette.muted;
  ctx.lineWidth = 1;
  if (horizontal) {
    const rowH = plotH / data.length;
    data.forEach((value, index) => {
      const y = pad.top + rowH * index + rowH * .25;
      const barW = (value / max) * plotW;
      ctx.textAlign = "right";
      ctx.fillText((labels || [])[index] || "", pad.left - 12, y + rowH * .25 + 4);
      ctx.fillStyle = chartPalette.canvas;
      ctx.beginPath();
      ctx.roundRect(pad.left, y, plotW, rowH * .5, 4);
      ctx.fill();
      ctx.fillStyle = palette[index % palette.length];
      ctx.beginPath();
      ctx.roundRect(pad.left, y, Math.max(4, barW), rowH * .5, 4);
      ctx.fill();
      ctx.fillStyle = chartPalette.text;
      ctx.textAlign = "left";
      ctx.fillText(`${Math.round(value)}`, pad.left + Math.max(8, barW) + 8, y + rowH * .25 + 4);
      ctx.fillStyle = chartPalette.muted;
    });
  } else {
    const gap = Math.max(12, plotW * .04);
    const barW = Math.max(22, (plotW - gap * (data.length - 1)) / data.length);
    data.forEach((value, index) => {
      const x = pad.left + index * (barW + gap);
      const barH = (value / max) * plotH;
      const y = pad.top + plotH - barH;
      ctx.fillStyle = chartPalette.canvas;
      ctx.fillRect(pad.left, pad.top, plotW, plotH);
      ctx.fillStyle = palette[index % palette.length];
      ctx.beginPath();
      ctx.roundRect(x, y, barW, Math.max(4, barH), 4);
      ctx.fill();
      ctx.fillStyle = chartPalette.text;
      ctx.textAlign = "center";
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      ctx.fillText(`${Math.round(value)}`, x + barW / 2, y - 8);
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.fillStyle = chartPalette.muted;
      ctx.fillText((labels || [])[index] || "", x + barW / 2, height - 14);
    });
  }
  return { fallback: true };
}

function drawDoughnutFallback(canvas, labels, values, colors) {
  const data = numericValues(values);
  const total = data.reduce((sum, value) => sum + value, 0);
  const prepared = prepareCanvas(canvas);
  if (!prepared) return null;
  const { ctx, width, height } = prepared;
  const palette = colors || [chartPalette.success, chartPalette.warning, chartPalette.danger, chartPalette.accent];
  const cx = width / 2;
  const cy = Math.max(104, height / 2 - 14);
  const radius = Math.min(width, height) * .27;
  const inner = radius * .68;
  let start = -Math.PI / 2;
  const safeTotal = total || 1;
  data.forEach((value, index) => {
    const angle = (value / safeTotal) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, start, start + angle);
    ctx.arc(cx, cy, inner, start + angle, start, true);
    ctx.closePath();
    ctx.fillStyle = total ? palette[index % palette.length] : chartPalette.border;
    ctx.fill();
    start += angle;
  });
  ctx.fillStyle = chartPalette.text;
  ctx.textAlign = "center";
  ctx.font = "700 24px Inter, system-ui, sans-serif";
  ctx.fillText(String(total), cx, cy + 6);
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillStyle = chartPalette.muted;
  ctx.fillText("items", cx, cy + 24);

  const legendY = height - 38;
  const legendLabels = labels || [];
  const itemW = Math.min(130, width / Math.max(1, legendLabels.length));
  legendLabels.forEach((label, index) => {
    const x = width / 2 - (itemW * legendLabels.length) / 2 + itemW * index + 10;
    ctx.fillStyle = palette[index % palette.length];
    ctx.beginPath();
    ctx.arc(x, legendY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = chartPalette.muted;
    ctx.textAlign = "left";
    ctx.fillText(label, x + 10, legendY + 4);
  });
  return { fallback: true };
}

function drawRadarFallback(canvas, labels, values) {
  const data = numericValues(values);
  if (!data.length) return drawFallbackEmpty(canvas);
  const prepared = prepareCanvas(canvas);
  if (!prepared) return null;
  const { ctx, width, height } = prepared;
  const cx = width / 2;
  const cy = height / 2 + 6;
  const radius = Math.min(width, height) * .32;
  const count = data.length;
  const angleFor = index => -Math.PI / 2 + (Math.PI * 2 * index) / count;

  ctx.strokeStyle = chartPalette.border;
  ctx.fillStyle = chartPalette.muted;
  ctx.textAlign = "center";
  for (let ring = 1; ring <= 4; ring += 1) {
    const r = radius * ring / 4;
    ctx.beginPath();
    for (let i = 0; i < count; i += 1) {
      const angle = angleFor(i);
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }
  for (let i = 0; i < count; i += 1) {
    const angle = angleFor(i);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    ctx.stroke();
    ctx.fillText((labels || [])[i] || "", cx + Math.cos(angle) * (radius + 24), cy + Math.sin(angle) * (radius + 24) + 4);
  }

  ctx.beginPath();
  data.forEach((value, index) => {
    const angle = angleFor(index);
    const r = radius * Math.max(0, Math.min(100, value)) / 100;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(25, 77, 58, 0.1)";
  ctx.strokeStyle = chartPalette.accent;
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
  return { fallback: true };
}

function makeLineChart(canvas, labels, values) {
  if (!canvas) return null;
  if (!window.Chart) return drawLineFallback(canvas, labels, values);
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, 260);
  gradient.addColorStop(0, "rgba(25, 77, 58, 0.12)");
  gradient.addColorStop(1, "rgba(25, 77, 58, 0)");
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
        pointRadius: 3,
        pointBackgroundColor: chartPalette.accent,
        pointBorderColor: chartPalette.white,
        pointHoverRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: chartPalette.text,
          titleColor: chartPalette.white,
          bodyColor: "#CBD5E1",
          borderColor: chartPalette.border,
          borderWidth: 1,
          cornerRadius: 6
        }
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: chartPalette.muted },
          grid: { color: chartPalette.border }
        },
        x: {
          ticks: { color: chartPalette.muted },
          grid: { display: false }
        }
      }
    }
  });
}

function makeHorizontalBarChart(canvas, labels, values, directions) {
  if (!canvas) return null;
  const colors = values.map((_, index) => directions[index] === "decreases_risk" ? chartPalette.success : chartPalette.danger);
  if (!window.Chart) return drawBarFallback(canvas, labels, values, colors, true);
  return new Chart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4 }] },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { color: chartPalette.muted }, grid: { color: chartPalette.border } },
        y: { ticks: { color: chartPalette.muted }, grid: { display: false } }
      }
    }
  });
}

function makeBarChart(canvas, labels, values, colors) {
  if (!canvas) return null;
  if (!window.Chart) return drawBarFallback(canvas, labels, values, colors, false);
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors || [chartPalette.accent, chartPalette.success, chartPalette.warning, chartPalette.info],
        borderRadius: 4,
        maxBarThickness: 42
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { color: chartPalette.muted }, grid: { color: chartPalette.border } },
        x: { ticks: { color: chartPalette.muted }, grid: { display: false } }
      }
    }
  });
}

function makeDoughnutChart(canvas, labels, values, colors) {
  if (!canvas) return null;
  if (!window.Chart) return drawDoughnutFallback(canvas, labels, values, colors);
  return new Chart(canvas, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors || [chartPalette.success, chartPalette.warning, chartPalette.danger, chartPalette.accent],
        borderColor: chartPalette.white,
        borderWidth: 3,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 10, usePointStyle: true, color: chartPalette.muted }
        }
      }
    }
  });
}

function makeRadarChart(canvas, labels, values) {
  if (!canvas) return null;
  if (!window.Chart) return drawRadarFallback(canvas, labels, values);
  return new Chart(canvas, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: chartPalette.accent,
        backgroundColor: "rgba(25, 77, 58, 0.1)",
        pointBackgroundColor: chartPalette.accent,
        pointBorderColor: chartPalette.white,
        pointRadius: 3,
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
