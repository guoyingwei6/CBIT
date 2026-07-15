import path from "node:path";
import { fileURLToPath } from "node:url";

import { Builder, Browser, By } from "selenium-webdriver";
import chrome from "selenium-webdriver/chrome.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argumentsByName = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  argumentsByName.set(process.argv[index], process.argv[index + 1]);
}
const url = argumentsByName.get("--url") ||
  "http://127.0.0.1:8080/#/GBC_estimation";
const genotypePath = path.resolve(
  argumentsByName.get("--genotype") ||
    path.join(ROOT, "public/examples/genotypes_for_GBC_extimator.txt")
);

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
  const analysisTab = await driver.findElement(
    By.xpath("//*[@role='tab' and normalize-space(.)='Analysis']")
  );
  await analysisTab.click();

  const input = await driver.findElement(By.css("input[type='file']"));
  await input.sendKeys(genotypePath);
  await driver.wait(
    async () =>
      (await driver.executeScript(
        "return document.querySelector('.upload-demo')?.textContent || ''"
      )).includes(path.basename(genotypePath)),
    10_000,
    "The selected GBC file was not shown in the UI"
  );

  await driver.executeScript(`
    window.__gbcMainRequests = [];
    const originalFetch = window.fetch;
    window.fetch = function(input, init) {
      const url = typeof input === "string" ? input : input.url;
      window.__gbcMainRequests.push(new URL(url, location.href).href);
      return originalFetch.call(this, input, init);
    };
  `);
  const analyseButton = await driver.findElement(
    By.xpath("//button[normalize-space(.)='Analyse']")
  );
  await analyseButton.click();

  await driver.wait(
    async () =>
      driver.executeScript(`
        const result = document.querySelector('a[download="CBIT-GBC-result.csv"]');
        const tip = document.querySelector('.tip span')?.textContent?.trim();
        return Boolean(result?.href?.startsWith('blob:') || tip);
      `),
    120_000,
    "The GBC UI analysis did not finish"
  );

  const state = await driver.executeScript(`
    const result = document.querySelector('a[download="CBIT-GBC-result.csv"]');
    return {
      tip: document.querySelector('.tip span')?.textContent?.trim() || '',
      downloadUrl: result?.href || '',
      visibleRows: document.querySelectorAll('.el-table__body-wrapper tbody tr').length,
      visibleColumns: document.querySelectorAll(
        '.el-table__header-wrapper th:not(.gutter)'
      ).length,
      cloudRequests: (window.__gbcMainRequests || []).filter((url) =>
        url.includes('/compute-api')
      ),
      unrelatedResources: performance.getEntriesByType('resource')
        .map((entry) => entry.name)
        .filter((name) => name.includes('/data/world.js') || name.includes('/images/')),
    };
  `);
  if (state.tip) {
    throw new Error(`GBC UI showed an error: ${state.tip}`);
  }
  if (!state.downloadUrl.startsWith("blob:")) {
    throw new Error("GBC UI did not create the result download");
  }
  if (state.visibleRows !== 10 || state.visibleColumns !== 33) {
    throw new Error(
      `Unexpected result table shape: ${state.visibleRows} rows x ` +
        `${state.visibleColumns} columns visible`
    );
  }
  if (state.cloudRequests.length > 0) {
    throw new Error(
      `Browser-first GBC unexpectedly called Cloud Run: ${state.cloudRequests.join(", ")}`
    );
  }
  if (state.unrelatedResources.length > 0) {
    throw new Error(`GBC route loaded home assets: ${state.unrelatedResources.join(", ")}`);
  }
  console.log(
    `GBC UI: pass, real file analysed locally, ` +
      `${state.visibleRows} result rows visible, download ready, 0 cloud requests`
  );
} finally {
  await driver.quit();
}
