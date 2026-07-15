const DEFAULT_MAX_SAMPLES = 100;
const DEFAULT_TIMEOUT_MS = 45_000;

function assetUrl(path) {
  return new URL(`/${path}`, window.location.origin).href;
}

export function supportsLocalGbc() {
  const mobile =
    (navigator.userAgentData && navigator.userAgentData.mobile) ||
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  return (
    !mobile &&
    (!navigator.deviceMemory || navigator.deviceMemory >= 4) &&
    typeof Worker !== "undefined" &&
    typeof WebAssembly !== "undefined" &&
    typeof DecompressionStream !== "undefined" &&
    typeof File !== "undefined" &&
    typeof File.prototype.stream === "function"
  );
}

export function analyzeGbcLocally(file, threshold, options = {}) {
  if (!supportsLocalGbc()) {
    return Promise.reject(
      Object.assign(new Error("Browser GBC analysis is unavailable"), {
        code: "LOCAL_UNAVAILABLE",
      })
    );
  }
  const maxSamples = options.maxSamples || DEFAULT_MAX_SAMPLES;
  const timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;
  let worker;
  try {
    worker = new Worker("/workers/gbc.worker.js", { type: "module" });
  } catch (error) {
    return Promise.reject(
      Object.assign(
        new Error(error.message || "Browser GBC worker could not start"),
        { code: "LOCAL_FAILURE" }
      )
    );
  }
  const requestId = 1;

  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = window.setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      worker.terminate();
      reject(
        Object.assign(new Error("Browser GBC analysis timed out"), {
          code: "LOCAL_UNAVAILABLE",
        })
      );
    }, timeoutMs);
    function finish() {
      if (settled) {
        return false;
      }
      settled = true;
      window.clearTimeout(timeout);
      worker.terminate();
      return true;
    }
    worker.addEventListener("message", ({ data }) => {
      if (data.requestId !== requestId) {
        return;
      }
      if (!finish()) {
        return;
      }
      if (data.error) {
        reject(Object.assign(new Error(data.error), { code: data.code }));
      } else {
        resolve(data.result);
      }
    });
    worker.addEventListener("error", (event) => {
      if (!finish()) {
        return;
      }
      reject(
        Object.assign(new Error(event.message || "Browser GBC worker failed"), {
          code: "LOCAL_FAILURE",
        })
      );
    });
    try {
      worker.postMessage({
        requestId,
        file,
        threshold,
        maxSamples,
        manifestUrl: assetUrl("gbc/gbc-reference.json"),
        kernelUrl: assetUrl("gbc/gbc-kernel.wasm"),
      });
    } catch (error) {
      if (finish()) {
        reject(
          Object.assign(
            new Error(error.message || "Browser GBC worker could not start"),
            { code: "LOCAL_FAILURE" }
          )
        );
      }
    }
  });
}
