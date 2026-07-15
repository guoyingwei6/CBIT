import * as ort from "../ort/ort.wasm.min.mjs";

const BATCH_SIZE = 256;
let metadataPromise;
let configuredWasmPath;
const sessions = new Map();

function parseLine(line, features, lineNumber) {
  const fields = line.trim().split(/\s+/);
  if (fields.length === 1 && fields[0] === "") {
    return null;
  }
  if (fields.length !== features + 1) {
    throw new Error(
      `Line ${lineNumber} has ${fields.length - 1} SNPs; the selected model requires ${features}`
    );
  }

  const values = new Float32Array(features);
  for (let index = 0; index < features; index += 1) {
    const rawValue = fields[index + 1];
    const normalized = rawValue.toUpperCase();
    if (normalized === "NA" || normalized === "NAN" || rawValue === ".") {
      values[index] = 0;
      continue;
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
      throw new Error(`Line ${lineNumber} contains an invalid genotype value`);
    }
    values[index] = value;
  }
  return { sample: fields[0], values };
}

async function loadMetadata(url) {
  if (!metadataPromise) {
    metadataPromise = fetch(url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load breed model metadata (${response.status})`);
        }
        return response.json();
      })
      .catch((error) => {
        metadataPromise = undefined;
        throw error;
      });
  }
  return metadataPromise;
}

async function loadSession(mode, modelUrl, wasmBaseUrl) {
  if (!configuredWasmPath) {
    configuredWasmPath = wasmBaseUrl;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = wasmBaseUrl;
  }
  if (configuredWasmPath !== wasmBaseUrl) {
    throw new Error("Breed model runtime path changed after initialization");
  }
  if (!sessions.has(mode)) {
    const sessionPromise = ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      })
      .catch((error) => {
        sessions.delete(mode);
        throw error;
      });
    sessions.set(mode, sessionPromise);
  }
  return sessions.get(mode);
}

async function inferBatch(session, rows, features, breeds, output) {
  if (rows.length === 0) {
    return;
  }
  const input = new Float32Array(rows.length * features);
  rows.forEach((row, index) => input.set(row.values, index * features));
  const result = await session.run({
    genotypes: new ort.Tensor("float32", input, [rows.length, features]),
  });
  const labels = result.label.data;
  const probabilities = result.probabilities.data;
  const classCount = probabilities.length / rows.length;

  rows.forEach((row, rowIndex) => {
    let maxProbability = -Infinity;
    const offset = rowIndex * classCount;
    for (let index = 0; index < classCount; index += 1) {
      maxProbability = Math.max(maxProbability, probabilities[offset + index]);
    }
    const code = String(Number(labels[rowIndex]));
    output.push({
      Sample: row.sample,
      Breed: breeds[code] || code,
      Probability: maxProbability,
    });
  });
}

function escapeCell(value) {
  const text = String(value);
  return /[\t\r\n"]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function toTsv(rows) {
  const columns = ["Sample", "Breed", "Probability"];
  return [
    columns.join("\t"),
    ...rows.map((row) => columns.map((column) => escapeCell(row[column])).join("\t")),
  ].join("\n");
}

async function analyze({ file, mode, metadataUrl, modelBaseUrl, wasmBaseUrl }) {
  const metadata = await loadMetadata(metadataUrl);
  const model = metadata.models[mode];
  if (!model) {
    throw new Error(`Unknown breed model: ${mode}`);
  }
  const session = await loadSession(mode, `${modelBaseUrl}${model.file}`, wasmBaseUrl);
  const reader = file.stream().getReader();
  const decoder = new TextDecoder();
  const rows = [];
  const output = [];
  let remainder = "";
  let lineNumber = 0;

  async function consume(line) {
    lineNumber += 1;
    const row = parseLine(line, model.features, lineNumber);
    if (row) {
      rows.push(row);
    }
    if (rows.length >= BATCH_SIZE) {
      await inferBatch(session, rows, model.features, metadata.breeds, output);
      rows.length = 0;
    }
  }

  let chunk = await reader.read();
  while (!chunk.done) {
    const { value } = chunk;
    remainder += decoder.decode(value, { stream: true });
    const lines = remainder.split(/\r\n|[\r\n]/);
    remainder = lines.pop();
    for (const line of lines) {
      await consume(line);
    }
    chunk = await reader.read();
  }
  remainder += decoder.decode();
  if (remainder.trim()) {
    await consume(remainder);
  }
  await inferBatch(session, rows, model.features, metadata.breeds, output);
  if (output.length === 0) {
    throw new Error("The genotype file contains no samples");
  }

  return {
    columns: ["Sample", "Breed", "Probability"],
    data: output,
    csv: toTsv(output),
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
    });
  }
});
