const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-web");

const MAX_PROBABILITY_ERROR = 1e-6;

async function verify(mode) {
  const fixturePath = path.resolve(__dirname, `fixtures/breed-${mode}.json`);
  const modelPath = path.resolve(
    __dirname,
    `../public/models/breed-${mode}.onnx`
  );
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  const model = new Uint8Array(fs.readFileSync(modelPath));
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
  });
  const result = await session.run({
    genotypes: new ort.Tensor(
      "float32",
      Float32Array.from(fixture.input),
      fixture.shape
    ),
  });
  const labels = Array.from(result.label.data, Number);
  const probabilities = Array.from(result.probabilities.data, Number);
  let maxProbabilityError = 0;

  labels.forEach((label, index) => {
    if (label !== fixture.labels[index]) {
      throw new Error(
        `${mode} label mismatch at row ${index}: ${label} != ${fixture.labels[index]}`
      );
    }
  });

  probabilities.forEach((probability, index) => {
    maxProbabilityError = Math.max(
      maxProbabilityError,
      Math.abs(probability - fixture.probabilities[index])
    );
  });

  if (maxProbabilityError > MAX_PROBABILITY_ERROR) {
    throw new Error(
      `${mode} maximum probability error ${maxProbabilityError} exceeds ${MAX_PROBABILITY_ERROR}`
    );
  }

  return { mode, rows: fixture.shape[0], maxProbabilityError };
}

(async () => {
  ort.env.wasm.numThreads = 1;
  for (const mode of ["fast", "accurate"]) {
    const result = await verify(mode);
    process.stdout.write(
      `${result.mode}: ${result.rows} rows, max probability error ${result.maxProbabilityError}\n`
    );
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
