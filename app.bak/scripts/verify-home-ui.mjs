import { Builder, Browser, By } from "selenium-webdriver";
import chrome from "selenium-webdriver/chrome.js";

const argumentsByName = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  argumentsByName.set(process.argv[index], process.argv[index + 1]);
}
const url =
  argumentsByName.get("--url") || "http://127.0.0.1:8080/#/home";

const options = new chrome.Options().addArguments(
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--window-size=1440,1200"
);
const driver = await new Builder()
  .forBrowser(Browser.CHROME)
  .setChromeOptions(options)
  .build();

try {
  await driver.manage().setTimeouts({ pageLoad: 120_000, script: 120_000 });
  await driver.get(url);
  await driver.wait(
    async () =>
      driver.executeScript(`
        const svg = document.querySelector('#canvas svg');
        return Boolean(svg && svg.querySelectorAll('path').length > 20);
      `),
    30_000,
    "The sample map did not render"
  );

  const inspectMap = async (label, width, height) => {
    await driver.manage().window().setRect({ width, height });
    await driver.sleep(200);
    const layout = await driver.executeScript(`
      const canvas = document.querySelector('#canvas');
      const canvasBox = canvas.getBoundingClientRect();
      const countryBoxes = [...document.querySelectorAll('.map-country')]
        .map((country) => country.getBoundingClientRect());
      const left = Math.min(...countryBoxes.map((box) => box.left));
      const top = Math.min(...countryBoxes.map((box) => box.top));
      const right = Math.max(...countryBoxes.map((box) => box.right));
      const bottom = Math.max(...countryBoxes.map((box) => box.bottom));
      return {
        canvasRatio: canvasBox.width / canvasBox.height,
        countryRatio: (right - left) / (bottom - top),
        verticalFill: (bottom - top) / canvasBox.height,
      };
    `);
    if (Math.abs(layout.canvasRatio - 2.4) > 0.03) {
      throw new Error(`${label} map container ratio is ${layout.canvasRatio.toFixed(3)}`);
    }
    if (layout.countryRatio < 2.5 || layout.countryRatio > 2.7) {
      throw new Error(`${label} world geometry ratio is ${layout.countryRatio.toFixed(3)}`);
    }
    if (layout.verticalFill < 0.85) {
      throw new Error(`${label} world map only fills ${(layout.verticalFill * 100).toFixed(1)}% vertically`);
    }
    return layout;
  };
  const portraitMap = await inspectMap("900x1600", 900, 1600);
  const landscapeMap = await inspectMap("1440x810", 1440, 810);

  const tableTab = await driver.findElement(
    By.xpath("//*[@role='tab' and normalize-space(.)='Sample info table']")
  );
  await tableTab.click();
  await driver.wait(
    async () =>
      driver.executeScript(
        "return document.querySelectorAll('.el-table__body-wrapper tbody tr').length === 10"
      ),
    10_000,
    "The sample table did not render"
  );

  const state = await driver.executeScript(`
    return {
      mapPaths: document.querySelectorAll('#canvas svg path').length,
      tableRows: document.querySelectorAll('.el-table__body-wrapper tbody tr').length,
      totalText: document.querySelector('.pagination')?.textContent || '',
      downloadUrl: document.querySelector('.table a')?.href || '',
      apiRequests: performance.getEntriesByType('resource')
        .map((entry) => entry.name)
        .filter((name) => name.includes('/api/mongo/')),
      pcaRequests: performance.getEntriesByType('resource')
        .map((entry) => entry.name)
        .filter((name) => name.endsWith('/pca.html') || name.endsWith('/data/pca.json')),
    };
  `);
  if (!state.totalText.includes("49")) {
    throw new Error(`Unexpected sample total: ${state.totalText.trim()}`);
  }
  if (!state.downloadUrl.endsWith("/data/sample_info.csv")) {
    throw new Error(`Unexpected sample download URL: ${state.downloadUrl}`);
  }
  if (state.apiRequests.length > 0) {
    throw new Error(
      `Home page still requested Django data: ${state.apiRequests.join(", ")}`
    );
  }
  if (state.pcaRequests.length > 0) {
    throw new Error(`PCA loaded before its tab was selected: ${state.pcaRequests.join(", ")}`);
  }

  const pcaTab = await driver.findElement(
    By.xpath("//*[@role='tab' and normalize-space(.)='PCA of all samples']")
  );
  await pcaTab.click();
  await driver.wait(
    async () =>
      driver.executeScript(`
        const frame = document.querySelector('.poa iframe');
        return Boolean(
          frame?.contentDocument?.querySelector('.plotly-graph-div .main-svg') &&
          frame.contentDocument.querySelectorAll('.legend-item').length === 49 &&
          frame.contentDocument.querySelector('.plotly-graph-div')?.dataset.samples === '2093' &&
          frame.contentDocument.querySelector('#pca-status')?.hidden
        );
      `),
    60_000,
    "The PCA plot did not render all breed traces"
  );
  console.log(
    `Home UI: pass, ${state.mapPaths} map paths, ${state.tableRows} visible rows, ` +
      `49 breed records, PCA ready, static download ready, 0 Django data requests; ` +
      `map ratios ${portraitMap.countryRatio.toFixed(3)} portrait / ` +
      `${landscapeMap.countryRatio.toFixed(3)} landscape`
  );
} finally {
  await driver.quit();
}
