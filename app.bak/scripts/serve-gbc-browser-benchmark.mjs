import { createReadStream, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argumentsByName = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  argumentsByName.set(process.argv[index], process.argv[index + 1]);
}
const genotypePath = path.resolve(
  argumentsByName.get("--genotype") || "/tmp/cbit-gbc-200k-100.txt"
);
const fixturePath = path.resolve(
  argumentsByName.get("--fixture") || "/tmp/cbit-gbc-200k-100.json"
);
const port = Number(argumentsByName.get("--port") || 4175);
const host = "127.0.0.1";
const manifest = JSON.parse(
  readFileSync(path.join(ROOT, "public/gbc/gbc-reference.json"), "utf8")
);

const page = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>CBIT GBC browser benchmark</title>
  </head>
  <body data-status="running">
    <h1>CBIT GBC browser benchmark</h1>
    <pre id="result">Running...</pre>
    <script type="module">
      const resultElement = document.querySelector("#result");
      const started = performance.now();
      try {
        const [fixtureResponse, expectedResponse] = await Promise.all([
          fetch("/fixture.txt"),
          fetch("/expected.json"),
        ]);
        const file = new File([await fixtureResponse.blob()], "benchmark.txt", {
          type: "text/plain",
        });
        const expected = await expectedResponse.json();
        const worker = new Worker("/gbc.worker.js", { type: "module" });
        const workerResult = await new Promise((resolve, reject) => {
          worker.addEventListener("message", ({ data }) => {
            worker.terminate();
            if (data.error) {
              reject(new Error(data.error));
            } else {
              resolve(data.result);
            }
          });
          worker.addEventListener("error", (event) => {
            worker.terminate();
            reject(new Error(event.message || "Worker failed"));
          });
          worker.postMessage({
            requestId: 1,
            file,
            threshold: expected.threshold,
            maxSamples: 100,
            manifestUrl: new URL("/gbc/gbc-reference.json", location.href).href,
            kernelUrl: new URL("/gbc/gbc-kernel.wasm", location.href).href,
          });
        });
        let mismatches = 0;
        let maximumDifference = 0;
        for (let row = 0; row < expected.data.length; row += 1) {
          for (const sample of expected.columns.slice(1)) {
            const expectedValue = expected.data[row][sample];
            const actualValue = workerResult.data[row][sample];
            if (!Object.is(expectedValue, actualValue)) {
              mismatches += 1;
              maximumDifference = Math.max(
                maximumDifference,
                Math.abs((expectedValue || 0) - (actualValue || 0))
              );
            }
          }
        }
        const result = {
          status: mismatches === 0 ? "pass" : "fail",
          mismatches,
          maximumDifference,
          totalSeconds: (performance.now() - started) / 1000,
          ...workerResult.metrics,
        };
        window.__gbcBenchmark = result;
        document.body.dataset.status = result.status;
        resultElement.textContent = JSON.stringify(result, null, 2);
      } catch (error) {
        const result = { status: "fail", error: error.message };
        window.__gbcBenchmark = result;
        document.body.dataset.status = "fail";
        resultElement.textContent = JSON.stringify(result, null, 2);
      }
    </script>
  </body>
</html>`;

const routes = new Map([
  ["/gbc.worker.js", path.join(ROOT, "public/workers/gbc.worker.js")],
  ["/gbc-core.mjs", path.join(ROOT, "public/workers/gbc-core.mjs")],
  ["/fixture.txt", genotypePath],
  ["/expected.json", fixturePath],
  ["/gbc/gbc-reference.json", path.join(ROOT, "public/gbc/gbc-reference.json")],
  ["/gbc/gbc-kernel.wasm", path.join(ROOT, "public/gbc/gbc-kernel.wasm")],
  [
    `/gbc/${manifest.file}`,
    path.join(ROOT, "public/gbc", manifest.file),
  ],
]);

function contentType(filename) {
  if (filename.endsWith(".wasm")) return "application/wasm";
  if (filename.endsWith(".json")) return "application/json; charset=utf-8";
  if (filename.endsWith(".js") || filename.endsWith(".mjs")) {
    return "text/javascript; charset=utf-8";
  }
  if (filename.endsWith(".gz")) return "application/gzip";
  return "text/plain; charset=utf-8";
}

const server = createServer((request, response) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  if (url.pathname === "/" || url.pathname === "/index.html") {
    response.writeHead(200, {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
    });
    response.end(page);
    return;
  }
  const filename = routes.get(url.pathname);
  if (!filename) {
    response.writeHead(404).end("Not found");
    return;
  }
  const size = statSync(filename).size;
  response.writeHead(200, {
    "Content-Type": contentType(filename),
    "Content-Length": size,
    "Cache-Control": "no-store",
  });
  createReadStream(filename).pipe(response);
});

server.listen(port, host, () => {
  console.log(`GBC browser benchmark: http://${host}:${port}/`);
});
