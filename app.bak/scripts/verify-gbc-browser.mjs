import { createReadStream, readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import path from "node:path";

import {
  BREED_COUNT,
  GbcAccumulator,
  normalise,
  parseDataLine,
  parseHeader,
  serialiseResult,
  solveLinearSystem,
} from "../public/workers/gbc-core.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argumentsByName = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  argumentsByName.set(process.argv[index], process.argv[index + 1]);
}
const manifestPath = path.join(ROOT, "public/gbc/gbc-reference.json");
const kernelPath = path.join(ROOT, "public/gbc/gbc-kernel.wasm");
const genotypePath = path.resolve(
  argumentsByName.get("--genotype") ||
    path.join(ROOT, "public/examples/genotypes_for_GBC_extimator.txt")
);
const fixturePath = path.resolve(
  argumentsByName.get("--fixture") || path.join(ROOT, "scripts/fixtures/gbc.json")
);
const maxSeconds = Number(argumentsByName.get("--max-seconds") || 30);
const maxMemoryMiB = Number(argumentsByName.get("--max-memory-mib") || 256);

function loadAccumulator(sampleCount) {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const module = new WebAssembly.Module(readFileSync(kernelPath));
  const wasm = new WebAssembly.Instance(module).exports;
  const accumulator = new GbcAccumulator(wasm, manifest, sampleCount);
  const compressed = readFileSync(path.join(path.dirname(manifestPath), manifest.file));
  const payload = gunzipSync(compressed);
  if (payload.byteLength !== manifest.uncompressedBytes) {
    throw new Error("Browser reference size does not match its manifest");
  }
  accumulator.payload.set(payload);
  accumulator.initializeReference();
  return { accumulator, manifest, wasm };
}

async function analyze() {
  const expected = JSON.parse(readFileSync(fixturePath, "utf8"));
  const input = createInterface({
    input: createReadStream(genotypePath),
    crlfDelay: Infinity,
  });
  let sampleNames;
  let runtime;
  let lineNumber = 0;
  const key = new Uint32Array(2);
  const started = performance.now();

  for await (const line of input) {
    if (!line.trim()) {
      continue;
    }
    lineNumber += 1;
    if (!sampleNames) {
      sampleNames = parseHeader(line, 1000);
      runtime = loadAccumulator(sampleNames.length);
      continue;
    }
    const { accumulator } = runtime;
    const usable = parseDataLine(
      line,
      sampleNames.length,
      accumulator.genotypes,
      accumulator.genotypeOffset(),
      key,
      lineNumber
    );
    if (!usable) {
      continue;
    }
    const referenceIndex = accumulator.matcher.find(key[0], key[1]);
    if (referenceIndex >= 0) {
      accumulator.commit(referenceIndex);
    }
  }

  runtime.accumulator.finish();
  const coefficients = solveLinearSystem(
    runtime.accumulator.gram,
    runtime.accumulator.cross,
    BREED_COUNT,
    sampleNames.length
  );
  const values = normalise(
    coefficients,
    expected.threshold,
    runtime.manifest.breeds,
    sampleNames
  );
  const actual = serialiseResult(runtime.manifest.breeds, sampleNames, values);
  const actualJson = JSON.stringify({
    matchedSnps: runtime.accumulator.matchedSnps,
    samples: sampleNames.length,
    columns: actual.columns,
    data: actual.data,
  });
  const expectedJson = JSON.stringify({
    matchedSnps: expected.matchedSnps,
    samples: expected.samples,
    columns: expected.columns,
    data: expected.data,
  });
  if (actualJson !== expectedJson) {
    let mismatches = 0;
    let maximumDifference = 0;
    for (let row = 0; row < expected.data.length; row += 1) {
      for (const sample of sampleNames) {
        const expectedValue = expected.data[row][sample];
        const actualValue = actual.data[row][sample];
        if (!Object.is(expectedValue, actualValue)) {
          mismatches += 1;
          maximumDifference = Math.max(
            maximumDifference,
            Math.abs((expectedValue || 0) - (actualValue || 0))
          );
        }
      }
    }
    throw new Error(
      `Browser GBC has ${mismatches} four-decimal mismatches; maximum difference ${maximumDifference}`
    );
  }
  const elapsedSeconds = (performance.now() - started) / 1000;
  const memoryMiB = runtime.wasm.memoryBytes() / (1024 * 1024);
  if (elapsedSeconds > maxSeconds) {
    throw new Error(
      `Browser GBC exceeded the ${maxSeconds}s target (${elapsedSeconds.toFixed(3)}s)`
    );
  }
  if (memoryMiB > maxMemoryMiB) {
    throw new Error(
      `Browser GBC exceeded the ${maxMemoryMiB} MiB target (${memoryMiB.toFixed(1)} MiB)`
    );
  }
  console.log(
    `Browser GBC: ${runtime.accumulator.matchedSnps} SNPs x ${sampleNames.length} samples, ` +
      `0 four-decimal mismatches, ${elapsedSeconds.toFixed(3)}s, ` +
      `${memoryMiB.toFixed(1)} MiB WASM memory`
  );
}

await analyze();
