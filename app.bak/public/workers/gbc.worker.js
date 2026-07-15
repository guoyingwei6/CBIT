import {
  BREED_COUNT,
  GbcAccumulator,
  GbcInputError,
  GbcLocalUnavailableError,
  normalise,
  parseDataLine,
  parseHeader,
  serialiseResult,
  solveLinearSystem,
} from "./gbc-core.mjs";

const MAX_HEADER_BYTES = 1024 * 1024;

async function loadHeader(file, maxSamples) {
  const prefix = await file.slice(0, MAX_HEADER_BYTES).text();
  const lines = prefix.split(/\r\n|[\r\n]/);
  for (const line of lines) {
    if (line.trim()) {
      return parseHeader(line, maxSamples);
    }
  }
  throw new GbcInputError(
    "The first row must start with CHR:POS and contain sample names"
  );
}

async function loadManifest(url) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new GbcLocalUnavailableError(
      `Unable to load the browser GBC reference (${response.status})`
    );
  }
  const manifest = await response.json();
  if (
    manifest.schemaVersion !== 1 ||
    manifest.valueDtype !== "float64" ||
    manifest.keyDtype !== "uint32-pair" ||
    !Array.isArray(manifest.breeds)
  ) {
    throw new GbcLocalUnavailableError("Unsupported browser GBC reference format");
  }
  return manifest;
}

async function instantiateKernel(url) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new GbcLocalUnavailableError(
      `Unable to load the browser GBC kernel (${response.status})`
    );
  }
  try {
    return (await WebAssembly.instantiateStreaming(response)).instance.exports;
  } catch (_error) {
    const fallback = await fetch(url, { cache: "force-cache" });
    if (!fallback.ok) {
      throw new GbcLocalUnavailableError(
        `Unable to load the browser GBC kernel (${fallback.status})`
      );
    }
    return (await WebAssembly.instantiate(await fallback.arrayBuffer())).instance
      .exports;
  }
}

async function loadReference(url, destination, expectedBytes) {
  if (typeof self.DecompressionStream === "undefined") {
    throw new GbcLocalUnavailableError(
      "This browser does not support streaming reference decompression"
    );
  }
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok || !response.body) {
    throw new GbcLocalUnavailableError(
      `Unable to load the browser GBC reference (${response.status})`
    );
  }
  const reader = response.body
    .pipeThrough(new self.DecompressionStream("gzip"))
    .getReader();
  let offset = 0;
  let chunk = await reader.read();
  while (!chunk.done) {
    const { value } = chunk;
    if (offset + value.byteLength > expectedBytes) {
      throw new GbcLocalUnavailableError("The browser GBC reference is too large");
    }
    destination.set(value, offset);
    offset += value.byteLength;
    chunk = await reader.read();
  }
  if (offset !== expectedBytes) {
    throw new GbcLocalUnavailableError("The browser GBC reference is incomplete");
  }
}

async function consumeFile(file, sampleNames, accumulator) {
  const reader = file.stream().getReader();
  const decoder = new TextDecoder();
  const key = new Uint32Array(2);
  let remainder = "";
  let lineNumber = 0;
  let headerSeen = false;

  function consumeLine(line) {
    if (!line.trim()) {
      return;
    }
    lineNumber += 1;
    if (!headerSeen) {
      const repeatedHeader = parseHeader(line, sampleNames.length);
      if (
        repeatedHeader.length !== sampleNames.length ||
        repeatedHeader.some((name, index) => name !== sampleNames[index])
      ) {
        throw new GbcInputError("The genotype header changed during analysis");
      }
      headerSeen = true;
      return;
    }

    const genotypeOffset = accumulator.genotypeOffset();
    const usable = parseDataLine(
      line,
      sampleNames.length,
      accumulator.genotypes,
      genotypeOffset,
      key,
      lineNumber
    );
    if (!usable) {
      return;
    }
    const referenceIndex = accumulator.matcher.find(key[0], key[1]);
    if (referenceIndex >= 0) {
      accumulator.commit(referenceIndex);
    }
  }

  let chunk = await reader.read();
  while (!chunk.done) {
    const { value } = chunk;
    remainder += decoder.decode(value, { stream: true });
    let start = 0;
    for (let index = 0; index < remainder.length; index += 1) {
      const code = remainder.charCodeAt(index);
      if (code === 10 || code === 13) {
        consumeLine(remainder.slice(start, index));
        if (code === 13 && remainder.charCodeAt(index + 1) === 10) {
          index += 1;
        }
        start = index + 1;
      }
    }
    remainder = remainder.slice(start);
    chunk = await reader.read();
  }
  remainder += decoder.decode();
  if (remainder.trim()) {
    consumeLine(remainder);
  }
  if (!headerSeen) {
    throw new GbcInputError(
      "The first row must start with CHR:POS and contain sample names"
    );
  }
}

async function analyze({
  file,
  threshold,
  manifestUrl,
  kernelUrl,
  maxSamples,
}) {
  const started = performance.now();
  const sampleNames = await loadHeader(file, maxSamples);
  const [manifest, wasm] = await Promise.all([
    loadManifest(manifestUrl),
    instantiateKernel(kernelUrl),
  ]);
  const accumulator = new GbcAccumulator(wasm, manifest, sampleNames.length);
  const referenceUrl = new URL(manifest.file, manifestUrl).href;
  await loadReference(
    referenceUrl,
    accumulator.payload,
    manifest.uncompressedBytes
  );
  accumulator.initializeReference();
  const computeStarted = performance.now();
  await consumeFile(file, sampleNames, accumulator);
  accumulator.finish();
  const coefficients = solveLinearSystem(
    accumulator.gram,
    accumulator.cross,
    BREED_COUNT,
    sampleNames.length
  );
  const values = normalise(
    coefficients,
    threshold,
    manifest.breeds,
    sampleNames
  );
  const result = serialiseResult(manifest.breeds, sampleNames, values);
  return {
    ...result,
    metrics: {
      matchedSnps: accumulator.matchedSnps,
      samples: sampleNames.length,
      elapsedSeconds: (performance.now() - started) / 1000,
      computeSeconds: (performance.now() - computeStarted) / 1000,
      wasmMemoryMiB: wasm.memoryBytes() / (1024 * 1024),
      execution: "browser",
    },
  };
}

self.addEventListener("message", async ({ data }) => {
  try {
    const result = await analyze(data);
    self.postMessage({ requestId: data.requestId, result });
  } catch (error) {
    self.postMessage({
      requestId: data.requestId,
      error: error instanceof Error ? error.message : String(error),
      code: error && error.code ? error.code : "LOCAL_FAILURE",
    });
  }
});
