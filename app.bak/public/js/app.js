import {
  analyzeBreedFile,
  releaseDownloadUrl,
  resultDownloadUrl,
} from "./services/breedAnalysis.js";
import {
  analyzeGbcFile,
  gbcResultDownloadUrl,
  releaseGbcDownloadUrl,
} from "./services/gbcAnalysis.js";

const ROUTES = new Set([
  "/home",
  "/Breed_identification",
  "/GBC_estimation",
  "/About",
]);
const SAMPLE_COLUMNS = [
  "Breed name",
  "Abbreviation",
  "Sample size",
  "Group",
  "Location",
  "Longitude (E)",
  "Latitude (N)",
];
const GROUP_COLORS = {
  NW: "red",
  EUR: "blue",
  TIB: "green",
  CEN: "yellow",
  SW: "purple",
  NE: "orange",
  SE: "pink",
  SOU: "#57e3eb",
  JP: "#3c8dbd",
  AMR: "#3c5488",
};
const BREED_IMAGES = [
  "zhoushan.jpg",
  "yunnangaofeng.jpg",
  "yanbian.jpg",
  "xizang.jpg",
  "wenshan.jpg",
  "wenlinggaofeng.jpg",
  "wannan.jpg",
  "Wagyu.jpg",
  "rikaze.jpg",
  "NorwayRed.jpg",
  "leiqionghainan.jpg",
  "jinnan.jpg",
  "jiangcheng.jpg",
  "Holstein.jpg",
  "hasake.jpg",
  "diqing.jpg",
  "dianzhong.jpg",
  "dengchuan.jpg",
  "BelgiumBlue.jpg",
  "apeijiaza.jpg",
];

const main = document.querySelector("#main-content");
const loading = document.querySelector("#loading");
let currentRoute = "/home";
let homePromise;
let carouselReady = false;
let breedDownloadUrl;
let gbcDownloadUrl;

function routeFromHash() {
  const route = location.hash.replace(/^#/, "") || "/home";
  return ROUTES.has(route) ? route : "/home";
}

function showRoute() {
  currentRoute = routeFromHash();
  const activePage = document.querySelector(`[data-route="${currentRoute}"]`);
  main.prepend(activePage);
  document.querySelectorAll("[data-route]").forEach((page) => {
    page.hidden = page !== activePage;
  });
  document.querySelectorAll("[data-route-link]").forEach((link) => {
    const active = link.dataset.routeLink === currentRoute;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.title = `${activePage.querySelector("h1")?.textContent.trim() || "CBIT"} | CBIT`;
  if (currentRoute === "/home") {
    loadHome();
    setupCarousel();
  }
  window.scrollTo({ top: 0, behavior: "instant" });
}

function setupTabs(group) {
  const buttons = [...group.querySelectorAll(":scope > .tab-list [data-tab]")];
  const panels = [...group.querySelectorAll(":scope > [data-panel]")];

  function activate(name) {
    buttons.forEach((button) => {
      const active = button.dataset.tab === name;
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.panel !== name;
    });
    if (name === "pca") {
      const frame = group.querySelector(".poa iframe");
      if (frame && !frame.src) {
        frame.src = frame.dataset.src;
      }
    }
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => activate(button.dataset.tab));
  });
  activate(buttons.find((button) => button.getAttribute("aria-selected") === "true")?.dataset.tab || buttons[0].dataset.tab);
}

function cellValue(row, column) {
  const value = row[column.key];
  return value === null || value === undefined ? "" : String(value);
}

function createDataTable(rootId, paginationId, initialPageSize = 10) {
  const root = document.querySelector(`#${rootId}`);
  const pagination = document.querySelector(`#${paginationId}`);
  let columns = [];
  let rows = [];
  let page = 1;
  let pageSize = initialPageSize;

  function normaliseColumns(values) {
    return values.map((column) =>
      typeof column === "string"
        ? { key: column, label: column }
        : { key: column.prop || column.key, label: column.label }
    );
  }

  function tableElement(kind, visibleRows = []) {
    const table = document.createElement("table");
    if (kind === "header") {
      const row = document.createElement("tr");
      columns.forEach((column) => {
        const cell = document.createElement("th");
        cell.scope = "col";
        cell.textContent = column.label;
        row.append(cell);
      });
      const head = document.createElement("thead");
      head.append(row);
      table.append(head);
      return table;
    }

    const body = document.createElement("tbody");
    visibleRows.forEach((item) => {
      const row = document.createElement("tr");
      columns.forEach((column) => {
        const cell = document.createElement("td");
        cell.textContent = cellValue(item, column);
        row.append(cell);
      });
      body.append(row);
    });
    table.append(body);
    return table;
  }

  function renderPagination() {
    const pages = Math.max(1, Math.ceil(rows.length / pageSize));
    page = Math.min(page, pages);
    pagination.replaceChildren();

    const total = document.createElement("strong");
    total.className = "total";
    total.textContent = `Totally ${rows.length} rows`;

    const size = document.createElement("select");
    size.setAttribute("aria-label", "Rows per page");
    [10, 20, 50, 100].forEach((value) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = `${value}rows/page`;
      option.selected = value === pageSize;
      size.append(option);
    });
    size.addEventListener("change", () => {
      pageSize = Number(size.value);
      page = 1;
      render();
    });

    const previous = document.createElement("button");
    previous.type = "button";
    previous.ariaLabel = "Previous page";
    previous.textContent = "‹";
    previous.disabled = page === 1;
    previous.addEventListener("click", () => {
      page -= 1;
      render();
    });

    const current = document.createElement("span");
    current.className = "current-page";
    current.textContent = String(page);

    const next = document.createElement("button");
    next.type = "button";
    next.ariaLabel = "Next page";
    next.textContent = "›";
    next.disabled = page === pages;
    next.addEventListener("click", () => {
      page += 1;
      render();
    });

    const gotoLabel = document.createElement("span");
    gotoLabel.textContent = "goto";
    const goto = document.createElement("input");
    goto.type = "number";
    goto.min = "1";
    goto.max = String(pages);
    goto.value = String(page);
    goto.setAttribute("aria-label", "Page number");
    goto.addEventListener("change", () => {
      page = Math.max(1, Math.min(pages, Number(goto.value) || 1));
      render();
    });
    const pageLabel = document.createElement("span");
    pageLabel.textContent = "page";

    pagination.append(total, size, previous, current, next, gotoLabel, goto, pageLabel);
  }

  function render() {
    const start = (page - 1) * pageSize;
    const visibleRows = rows.slice(start, start + pageSize);
    const header = document.createElement("div");
    header.className = "el-table__header-wrapper";
    header.append(tableElement("header"));
    const body = document.createElement("div");
    body.className = "el-table__body-wrapper";
    if (visibleRows.length) {
      body.append(tableElement("body", visibleRows));
    } else {
      const empty = document.createElement("div");
      empty.className = "empty-table";
      empty.textContent = "No data";
      body.append(empty);
    }
    body.addEventListener("scroll", () => {
      header.scrollLeft = body.scrollLeft;
    });
    root.replaceChildren(header, body);
    renderPagination();
  }

  return {
    setData(nextColumns, nextRows) {
      columns = normaliseColumns(nextColumns);
      rows = nextRows;
      page = 1;
      render();
    },
  };
}

function parseSampleInfo(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const headers = lines.shift().split("\t").map((value) => value.replace(/\u00a0/g, "").trim());
  return lines.filter(Boolean).map((line) => {
    const fields = line.split("\t");
    const row = Object.fromEntries(headers.map((header, index) => [header, fields[index]?.trim() || ""]));
    row["Sample size"] = Number(row["Sample size"]);
    row["Longitude (E)"] = Number(row["Longitude (E)"]);
    row["Latitude (N)"] = Number(row["Latitude (N)"]);
    return row;
  });
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function visitCoordinates(coordinates, callback) {
  if (typeof coordinates?.[0] === "number") {
    callback(coordinates);
    return;
  }
  coordinates?.forEach((child) => visitCoordinates(child, callback));
}

function createMapProjection(world, width, height, padding = 20) {
  const bounds = {
    minLongitude: Infinity,
    maxLongitude: -Infinity,
    minLatitude: Infinity,
    maxLatitude: -Infinity,
  };
  world.features.forEach((feature) => {
    visitCoordinates(feature.geometry?.coordinates, ([longitude, latitude]) => {
      bounds.minLongitude = Math.min(bounds.minLongitude, longitude);
      bounds.maxLongitude = Math.max(bounds.maxLongitude, longitude);
      bounds.minLatitude = Math.min(bounds.minLatitude, latitude);
      bounds.maxLatitude = Math.max(bounds.maxLatitude, latitude);
    });
  });

  const longitudeSpan = bounds.maxLongitude - bounds.minLongitude;
  const latitudeSpan = bounds.maxLatitude - bounds.minLatitude;
  const scale = Math.min(
    (width - padding * 2) / longitudeSpan,
    (height - padding * 2) / latitudeSpan
  );
  const mapWidth = longitudeSpan * scale;
  const mapHeight = latitudeSpan * scale;
  const offsetX = (width - mapWidth) / 2 - bounds.minLongitude * scale;
  const offsetY = (height - mapHeight) / 2 + bounds.maxLatitude * scale;

  return ([longitude, latitude]) => [
    offsetX + Number(longitude) * scale,
    offsetY - Number(latitude) * scale,
  ];
}

function geometryPath(geometry, projectCoordinate) {
  const polygons = geometry.type === "Polygon"
    ? [geometry.coordinates]
    : geometry.type === "MultiPolygon"
      ? geometry.coordinates
      : [];
  return polygons.map((polygon) =>
    polygon.map((ring) =>
      ring.map((coordinate, index) => {
        const [x, y] = projectCoordinate(coordinate);
        return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
      }).join("") + "Z"
    ).join("")
  ).join("");
}

function renderMap(rows, world) {
  const width = 1200;
  const height = 500;
  const canvas = document.querySelector("#canvas");
  const tooltip = document.querySelector("#map-tooltip");
  const projectCoordinate = createMapProjection(world, width, height);
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
  });
  const viewport = svgElement("g");
  const countries = svgElement("g");
  const sites = svgElement("g");

  world.features.forEach((feature) => {
    const pathData = geometryPath(feature.geometry, projectCoordinate);
    if (!pathData) return;
    const path = svgElement("path", { d: pathData, class: "map-country" });
    countries.append(path);
  });

  rows.forEach((row) => {
    const [cx, cy] = projectCoordinate([row["Longitude (E)"], row["Latitude (N)"]]);
    const circle = svgElement("circle", {
      cx,
      cy,
      r: Math.max(3, row["Sample size"] / 10),
      fill: GROUP_COLORS[row.Group] || "#666",
      class: "sample-site",
      tabindex: "0",
    });
    const showTooltip = (event) => {
      tooltip.innerHTML = `<strong>${row["Breed name"]}</strong><br>Breed name: ${row["Breed name"]}<br>Group: ${row.Group}<br>Location: ${row.Location}<br>Sample size: ${row["Sample size"]}`;
      tooltip.hidden = false;
      tooltip.style.left = `${event.clientX + 12}px`;
      tooltip.style.top = `${event.clientY + 12}px`;
    };
    circle.addEventListener("pointerenter", showTooltip);
    circle.addEventListener("pointermove", showTooltip);
    circle.addEventListener("pointerleave", () => { tooltip.hidden = true; });
    circle.addEventListener("blur", () => { tooltip.hidden = true; });
    circle.addEventListener("focus", () => {
      const box = circle.getBoundingClientRect();
      showTooltip({ clientX: box.left, clientY: box.top });
    });
    sites.append(circle);
  });

  viewport.append(countries, sites);
  svg.append(viewport);
  canvas.replaceChildren(svg);

  const transform = { x: 0, y: 0, scale: 1 };
  const updateTransform = () => {
    viewport.setAttribute("transform", `translate(${transform.x} ${transform.y}) scale(${transform.scale})`);
  };
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const bounds = svg.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * width;
    const y = ((event.clientY - bounds.top) / bounds.height) * height;
    const nextScale = Math.max(1, Math.min(8, transform.scale * Math.exp(-event.deltaY * 0.0015)));
    transform.x = x - ((x - transform.x) * nextScale) / transform.scale;
    transform.y = y - ((y - transform.y) * nextScale) / transform.scale;
    transform.scale = nextScale;
    updateTransform();
  }, { passive: false });

  let drag;
  svg.addEventListener("pointerdown", (event) => {
    const bounds = svg.getBoundingClientRect();
    drag = { x: event.clientX, y: event.clientY, sx: width / bounds.width, sy: height / bounds.height };
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    transform.x += (event.clientX - drag.x) * drag.sx;
    transform.y += (event.clientY - drag.y) * drag.sy;
    drag.x = event.clientX;
    drag.y = event.clientY;
    updateTransform();
  });
  const endDrag = () => { drag = undefined; };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
}

const sampleTable = createDataTable("sample-table", "sample-pagination");
let allSampleRows = [];

async function loadHome() {
  if (homePromise) return homePromise;
  homePromise = Promise.all([
    import("/data/world.js"),
    fetch("/data/sample_info.csv", { cache: "force-cache" }),
  ])
    .then(([worldModule, response]) => {
      if (!response.ok) throw new Error(`Sample data request failed (${response.status})`);
      return Promise.all([worldModule.default, response.text()]);
    })
    .then(([world, text]) => {
      allSampleRows = parseSampleInfo(text);
      sampleTable.setData(SAMPLE_COLUMNS, allSampleRows);
      renderMap(allSampleRows, world);
    })
    .catch((error) => {
      document.querySelector("#canvas").textContent = error.message;
      throw error;
    });
  return homePromise;
}

function setupCarousel() {
  if (carouselReady) return;
  carouselReady = true;
  const image = document.querySelector("#breed-photo");
  let index = 0;
  const show = (nextIndex) => {
    index = (nextIndex + BREED_IMAGES.length) % BREED_IMAGES.length;
    const name = BREED_IMAGES[index];
    image.src = `/images/${name}`;
    image.alt = name.replace(/\.[^.]+$/, "").replace(/([a-z])([A-Z])/g, "$1 $2");
  };
  document.querySelector(".carousel-control.previous").addEventListener("click", () => show(index - 1));
  document.querySelector(".carousel-control.next").addEventListener("click", () => show(index + 1));
  window.setInterval(() => {
    if (currentRoute === "/home" && !document.hidden) show(index + 1);
  }, 5000);
  show(0);
}

function setupFilePicker(inputId, statusId) {
  const input = document.querySelector(`#${inputId}`);
  const dropzone = input.closest(".upload-demo");
  const status = document.querySelector(`#${statusId}`);
  const title = dropzone.querySelector(".upload-copy strong");
  const initialTitle = title.textContent;
  const listeners = [];
  let file;

  function select(nextFile) {
    file = nextFile;
    title.textContent = file ? file.name : initialTitle;
    status.hidden = !file;
    status.textContent = file ? `${file.name} is ready for analysis.` : "";
    listeners.forEach((listener) => listener(file));
  }

  input.addEventListener("change", () => select(input.files[0]));
  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  });
  dropzone.addEventListener("drop", (event) => select(event.dataTransfer.files[0]));

  return {
    get file() { return file; },
    onChange(callback) { listeners.push(callback); },
  };
}

function setTip(id, message = "") {
  const tip = document.querySelector(`#${id}`);
  tip.hidden = !message;
  tip.querySelector("span").textContent = message;
}

function setLoading(active) {
  loading.hidden = !active;
  document.body.setAttribute("aria-busy", String(active));
}

const breedTable = createDataTable("breed-table", "breed-pagination");
const breedFile = setupFilePicker("breed-file", "breed-file-status");
breedFile.onChange(() => {
  setTip("breed-tip");
  document.querySelector("#breed-results").hidden = true;
  releaseDownloadUrl(breedDownloadUrl);
  breedDownloadUrl = undefined;
});

document.querySelector("#breed-analyse").addEventListener("click", async () => {
  if (!breedFile.file) {
    setTip("breed-tip", "Please choose a genotype file");
    return;
  }
  setTip("breed-tip");
  setLoading(true);
  try {
    const mode = document.querySelector("#breed-model").value;
    const result = await analyzeBreedFile(breedFile.file, mode);
    breedTable.setData(result.columns, result.data);
    releaseDownloadUrl(breedDownloadUrl);
    breedDownloadUrl = resultDownloadUrl(result.csv);
    document.querySelector("#breed-download").href = breedDownloadUrl;
    document.querySelector("#breed-results").hidden = false;
  } catch (error) {
    setTip("breed-tip", error.message);
  } finally {
    setLoading(false);
  }
});

const gbcTable = createDataTable("gbc-table", "gbc-pagination");
const gbcFile = setupFilePicker("gbc-file", "gbc-file-status");
gbcFile.onChange(() => {
  setTip("gbc-tip");
  document.querySelector("#gbc-results").hidden = true;
  releaseGbcDownloadUrl(gbcDownloadUrl);
  gbcDownloadUrl = undefined;
});

document.querySelector("#gbc-analyse").addEventListener("click", async () => {
  if (!gbcFile.file) {
    setTip("gbc-tip", "Please choose a genotype file");
    return;
  }
  const threshold = Number(document.querySelector("#gbc-threshold").value);
  if (!Number.isFinite(threshold) || threshold < 0.01 || threshold > 1) {
    setTip("gbc-tip", "The confidence threshold must be between 0.01 and 1");
    return;
  }
  setTip("gbc-tip");
  setLoading(true);
  try {
    const result = await analyzeGbcFile(gbcFile.file, threshold);
    const columns = result.columns.map((column) => ({
      prop: column,
      label: column === "Unnamed: 0" ? "Location" : column,
    }));
    gbcTable.setData(columns, result.data);
    releaseGbcDownloadUrl(gbcDownloadUrl);
    gbcDownloadUrl = gbcResultDownloadUrl(result.csv);
    document.querySelector("#gbc-download").href = gbcDownloadUrl;
    document.querySelector("#gbc-results").hidden = false;
  } catch (error) {
    setTip("gbc-tip", error.message);
  } finally {
    setLoading(false);
  }
});

document.querySelector("#sample-filter").addEventListener("input", (event) => {
  const query = event.target.value.trim().toLowerCase();
  const rows = query
    ? allSampleRows.filter((row) => SAMPLE_COLUMNS.some((column) => String(row[column]).toLowerCase().includes(query)))
    : allSampleRows;
  sampleTable.setData(SAMPLE_COLUMNS, rows);
});

document.querySelectorAll("[data-tabs]").forEach(setupTabs);
window.addEventListener("hashchange", showRoute);
window.addEventListener("beforeunload", () => {
  releaseDownloadUrl(breedDownloadUrl);
  releaseGbcDownloadUrl(gbcDownloadUrl);
});

if (!ROUTES.has(location.hash.replace(/^#/, ""))) {
  location.hash = "#/home";
}
showRoute();
