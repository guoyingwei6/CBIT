import { Builder, Browser } from "selenium-webdriver";
import chrome from "selenium-webdriver/chrome.js";
import edge from "selenium-webdriver/edge.js";
import firefox from "selenium-webdriver/firefox.js";

const argumentsByName = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  argumentsByName.set(process.argv[index], process.argv[index + 1]);
}
const url = argumentsByName.get("--url") || "http://127.0.0.1:4175/";
const requested = (argumentsByName.get("--browsers") || "chrome,edge,firefox")
  .split(",")
  .map((value) => value.trim().toLowerCase())
  .filter(Boolean);
const maxSeconds = Number(argumentsByName.get("--max-seconds") || 30);
const maxMemoryMiB = Number(argumentsByName.get("--max-memory-mib") || 256);

function builderFor(name) {
  if (name === "chrome") {
    const options = new chrome.Options().addArguments(
      "--headless=new",
      "--disable-gpu",
      "--no-sandbox"
    );
    return new Builder().forBrowser(Browser.CHROME).setChromeOptions(options);
  }
  if (name === "edge") {
    const options = new edge.Options().addArguments(
      "--headless=new",
      "--disable-gpu",
      "--no-sandbox"
    );
    return new Builder().forBrowser(Browser.EDGE).setEdgeOptions(options);
  }
  if (name === "firefox") {
    const options = new firefox.Options().addArguments("-headless");
    return new Builder().forBrowser(Browser.FIREFOX).setFirefoxOptions(options);
  }
  if (name === "safari") {
    return new Builder().forBrowser(Browser.SAFARI);
  }
  throw new Error(`Unknown browser: ${name}`);
}

async function run(name) {
  const driver = await builderFor(name).build();
  try {
    await driver.manage().setTimeouts({ pageLoad: 120_000, script: 120_000 });
    await driver.get(url);
    await driver.wait(
      async () =>
        (await driver.executeScript("return document.body.dataset.status")) !==
        "running",
      120_000,
      "GBC browser benchmark did not finish",
      100
    );
    const result = await driver.executeScript("return window.__gbcBenchmark");
    if (!result || result.status !== "pass") {
      throw new Error(JSON.stringify(result || { status: "missing" }));
    }
    if (result.elapsedSeconds > maxSeconds) {
      throw new Error(
        `worker time ${result.elapsedSeconds.toFixed(3)}s exceeds ${maxSeconds}s`
      );
    }
    if (result.wasmMemoryMiB > maxMemoryMiB) {
      throw new Error(
        `WASM memory ${result.wasmMemoryMiB.toFixed(1)} MiB exceeds ${maxMemoryMiB} MiB`
      );
    }
    return result;
  } finally {
    await driver.quit();
  }
}

let failed = false;
for (const name of requested) {
  try {
    const result = await run(name);
    console.log(
      `${name}: pass, 0 mismatches, worker ${result.elapsedSeconds.toFixed(3)}s, ` +
        `compute ${result.computeSeconds.toFixed(3)}s, ` +
        `${result.wasmMemoryMiB.toFixed(1)} MiB WASM memory`
    );
  } catch (error) {
    failed = true;
    console.error(`${name}: fail, ${error.message}`);
  }
}
if (failed) {
  process.exitCode = 1;
}
