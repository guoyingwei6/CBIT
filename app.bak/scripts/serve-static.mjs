import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argumentsByName = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  argumentsByName.set(process.argv[index], process.argv[index + 1]);
}
const root = path.resolve(project, argumentsByName.get("--root") || "public");
const host = argumentsByName.get("--host") || "127.0.0.1";
const port = Number(argumentsByName.get("--port") || 8080);
const contentTypes = new Map([
  [".bin", "application/octet-stream"],
  [".css", "text/css; charset=utf-8"],
  [".csv", "text/csv; charset=utf-8"],
  [".gz", "application/gzip"],
  [".html", "text/html; charset=utf-8"],
  [".ico", "image/x-icon"],
  [".jpg", "image/jpeg"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".png", "image/png"],
  [".txt", "text/plain; charset=utf-8"],
  [".wasm", "application/wasm"],
]);

createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, `http://${request.headers.host}`).pathname);
  const relative = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
  const filename = path.resolve(root, relative);
  if (!filename.startsWith(`${root}${path.sep}`)) {
    response.writeHead(403).end("Forbidden");
    return;
  }
  try {
    const stat = statSync(filename);
    if (!stat.isFile()) throw new Error("Not a file");
    response.writeHead(200, {
      "Content-Type": contentTypes.get(path.extname(filename).toLowerCase()) || "application/octet-stream",
      "Content-Length": stat.size,
      "Cache-Control": "no-store",
    });
    createReadStream(filename).pipe(response);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }).end("Not found");
  }
}).listen(port, host, () => {
  console.log(`CBIT: http://${host}:${port}/#/home`);
});

