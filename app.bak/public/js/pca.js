const viewer = document.querySelector("#pca-viewer");
const plotArea = viewer.querySelector(".plot-area");
const canvas = viewer.querySelector("canvas");
const context = canvas.getContext("2d", { alpha: false });
const overlay = viewer.querySelector(".scene-overlay");
const tooltip = document.querySelector("#pca-tooltip");
const status = document.querySelector("#pca-status");
const legend = viewer.querySelector(".legend-list");
const initialView = { yaw: -0.7, pitch: 0.48, zoom: 1 };
const view = { ...initialView };
let traces = [];
let points = [];
let projected = [];
let width = 1;
let height = 1;
let drag;
let frame;

function rotate(point) {
  const cosineY = Math.cos(view.yaw);
  const sineY = Math.sin(view.yaw);
  const x = point.x * cosineY - point.z * sineY;
  const z = point.x * sineY + point.z * cosineY;
  const cosineX = Math.cos(view.pitch);
  const sineX = Math.sin(view.pitch);
  return {
    x,
    y: point.y * cosineX - z * sineX,
    z: point.y * sineX + z * cosineX,
  };
}

function project(point) {
  const rotated = rotate(point);
  const perspective = 3.2 / (3.2 - rotated.z * 0.65);
  const scale = Math.min(width, height) * 0.36 * view.zoom * perspective;
  return {
    x: width / 2 + rotated.x * scale,
    y: height / 2 - rotated.y * scale,
    depth: rotated.z,
  };
}

function drawAxes() {
  const origin = project({ x: 0, y: 0, z: 0 });
  const axes = [
    { point: { x: 1.08, y: 0, z: 0 }, label: "PC 1", color: "#64748b" },
    { point: { x: 0, y: 1.08, z: 0 }, label: "PC 2", color: "#64748b" },
    { point: { x: 0, y: 0, z: 1.08 }, label: "PC 3", color: "#64748b" },
  ];
  context.lineWidth = 1.2;
  context.font = "12px Arial";
  axes.forEach((axis) => {
    const end = project(axis.point);
    context.strokeStyle = axis.color;
    context.beginPath();
    context.moveTo(origin.x, origin.y);
    context.lineTo(end.x, end.y);
    context.stroke();
    context.fillStyle = "#2a3f5f";
    context.fillText(axis.label, end.x + 5, end.y - 4);
  });
}

function render() {
  frame = undefined;
  context.fillStyle = "#e5ecf6";
  context.fillRect(0, 0, width, height);
  drawAxes();
  projected = points
    .filter((point) => traces[point.trace].visible)
    .map((point) => ({ ...project(point), point }))
    .sort((left, right) => left.depth - right.depth);
  projected.forEach((item) => {
    context.globalAlpha = 0.84;
    context.fillStyle = traces[item.point.trace].color;
    context.beginPath();
    context.arc(item.x, item.y, 3.4, 0, Math.PI * 2);
    context.fill();
  });
  context.globalAlpha = 1;
}

function requestRender() {
  if (!frame) frame = requestAnimationFrame(render);
}

function resize() {
  const bounds = plotArea.getBoundingClientRect();
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  width = Math.max(1, Math.round(bounds.width));
  height = Math.max(1, Math.round(bounds.height));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  requestRender();
}

function showTooltip(event) {
  let nearest;
  let distance = 64;
  projected.forEach((item) => {
    const current = (item.x - event.offsetX) ** 2 + (item.y - event.offsetY) ** 2;
    if (current < distance) {
      distance = current;
      nearest = item.point;
    }
  });
  if (!nearest) {
    tooltip.hidden = true;
    return;
  }
  const trace = traces[nearest.trace];
  tooltip.innerHTML = `<strong>${trace.name}</strong><br>SM=${nearest.id}<br>PC 1=${nearest.rawX}<br>PC 2=${nearest.rawY}<br>PC 3=${nearest.rawZ}`;
  tooltip.hidden = false;
  tooltip.style.left = `${Math.min(width - 220, event.offsetX + 12)}px`;
  tooltip.style.top = `${Math.max(8, event.offsetY - 24)}px`;
}

function buildLegend() {
  traces.forEach((trace, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "legend-item";
    button.title = trace.name;
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = trace.color;
    const name = document.createElement("span");
    name.className = "legend-name";
    name.textContent = trace.name;
    button.append(swatch, name);
    button.addEventListener("click", () => {
      trace.visible = !trace.visible;
      button.classList.toggle("disabled", !trace.visible);
      requestRender();
    });
    legend.append(button);
  });
}

canvas.addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, y: event.clientY };
  canvas.setPointerCapture(event.pointerId);
  tooltip.hidden = true;
});
canvas.addEventListener("pointermove", (event) => {
  if (!drag) {
    showTooltip(event);
    return;
  }
  view.yaw += (event.clientX - drag.x) * 0.008;
  view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (event.clientY - drag.y) * 0.008));
  drag.x = event.clientX;
  drag.y = event.clientY;
  requestRender();
});
canvas.addEventListener("pointerleave", () => { if (!drag) tooltip.hidden = true; });
canvas.addEventListener("pointerup", () => { drag = undefined; });
canvas.addEventListener("pointercancel", () => { drag = undefined; });
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  view.zoom = Math.max(0.55, Math.min(3.2, view.zoom * Math.exp(-event.deltaY * 0.0015)));
  requestRender();
}, { passive: false });

document.querySelector("#reset-view").addEventListener("click", () => {
  Object.assign(view, initialView);
  requestRender();
});
document.querySelector("#save-image").addEventListener("click", () => {
  canvas.toBlob((blob) => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "CBIT-PCA.png";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 0);
  }, "image/png");
});

try {
  const response = await fetch("/data/pca.json", { cache: "force-cache" });
  if (!response.ok) throw new Error(`PCA data request failed (${response.status})`);
  const data = await response.json();
  const values = data.traces.flatMap((trace) => trace.x.map((x, index) => ({
    x,
    y: trace.y[index],
    z: trace.z[index],
  })));
  const limits = ["x", "y", "z"].map((axis) => {
    const axisValues = values.map((point) => point[axis]);
    return { min: Math.min(...axisValues), max: Math.max(...axisValues) };
  });
  const centers = limits.map((limit) => (limit.min + limit.max) / 2);
  const span = Math.max(...limits.map((limit) => limit.max - limit.min));
  traces = data.traces.map((trace) => ({ ...trace, visible: true }));
  points = traces.flatMap((trace, traceIndex) => trace.x.map((rawX, index) => ({
    trace: traceIndex,
    id: trace.ids[index],
    rawX,
    rawY: trace.y[index],
    rawZ: trace.z[index],
    x: ((rawX - centers[0]) / span) * 2,
    y: ((trace.y[index] - centers[1]) / span) * 2,
    z: ((trace.z[index] - centers[2]) / span) * 2,
  })));
  viewer.dataset.samples = String(points.length);
  buildLegend();
  status.hidden = true;
  overlay.classList.add("main-svg");
  new ResizeObserver(resize).observe(plotArea);
  resize();
} catch (error) {
  status.textContent = error.message;
}
