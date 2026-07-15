const MAX_FILE_SIZE = 500 * 1024 * 1024;

let worker;
let nextRequestId = 1;
const pendingRequests = new Map();

function assetUrl(path) {
  return new URL(`/${path}`, window.location.origin).href;
}

function getWorker() {
  if (worker) {
    return worker;
  }

  worker = new Worker("/workers/breed.worker.js", { type: "module" });
  worker.addEventListener("message", ({ data }) => {
    const pending = pendingRequests.get(data.requestId);
    if (!pending) {
      return;
    }
    pendingRequests.delete(data.requestId);
    if (data.error) {
      pending.reject(new Error(data.error));
    } else {
      pending.resolve(data.result);
    }
  });
  worker.addEventListener("error", (event) => {
    const error = new Error(event.message || "Breed analysis worker failed");
    pendingRequests.forEach(({ reject }) => reject(error));
    pendingRequests.clear();
    worker.terminate();
    worker = undefined;
  });
  return worker;
}

export function analyzeBreedFile(file, mode) {
  if (!(file instanceof File)) {
    return Promise.reject(new Error("Please choose a genotype file"));
  }
  if (file.size > MAX_FILE_SIZE) {
    return Promise.reject(new Error("The genotype file exceeds the 500MB limit"));
  }
  if (!["fast", "accurate"].includes(mode)) {
    return Promise.reject(new Error(`Unknown breed model: ${mode}`));
  }

  const requestId = nextRequestId++;
  const request = new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });
  });
  getWorker().postMessage({
    requestId,
    file,
    mode,
    metadataUrl: assetUrl("models/breed-models.json"),
    modelBaseUrl: assetUrl("models/"),
    wasmBaseUrl: assetUrl("ort/"),
  });
  return request;
}

export function resultDownloadUrl(csv) {
  return URL.createObjectURL(
    new Blob([csv], { type: "text/tab-separated-values;charset=utf-8" })
  );
}

export function releaseDownloadUrl(url) {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}
