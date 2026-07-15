const fs = require("fs");
const path = require("path");

const sourceDir = path.dirname(
  require.resolve("onnxruntime-web/ort-wasm-simd-threaded.mjs")
);
const outputDir = path.resolve(__dirname, "../public/ort");
const assets = [
  "ort.wasm.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
];

fs.mkdirSync(outputDir, { recursive: true });

for (const asset of assets) {
  fs.copyFileSync(path.join(sourceDir, asset), path.join(outputDir, asset));
}
