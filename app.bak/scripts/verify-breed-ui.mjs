import path from "node:path";
import { fileURLToPath } from "node:url";

import { Builder, Browser, By } from "selenium-webdriver";
import chrome from "selenium-webdriver/chrome.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argumentsByName = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  argumentsByName.set(process.argv[index], process.argv[index + 1]);
}
const url = argumentsByName.get("--url") || "http://127.0.0.1:8080/#/Breed_identification";
const genotypePath = path.resolve(
  argumentsByName.get("--genotype") ||
    path.join(root, "public/examples/genotypes_for_Breed_identifier_fast_model.txt")
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
  await driver.findElement(
    By.xpath("//*[@role='tab' and normalize-space(.)='Analysis']")
  ).click();
  await driver.findElement(By.css("input[type='file']")).sendKeys(genotypePath);
  await driver.wait(
    async () => (await driver.executeScript(
      "return document.querySelector('.upload-demo')?.textContent || ''"
    )).includes(path.basename(genotypePath)),
    10_000,
    "The selected breed file was not shown in the UI"
  );
  await driver.findElement(By.xpath("//button[normalize-space(.)='Analyse']")).click();
  await driver.wait(
    async () => driver.executeScript(`
      const result = document.querySelector('a[download="CBIT-breed-identification.tsv"]');
      const tip = document.querySelector('.tip span')?.textContent?.trim();
      return Boolean(result?.href?.startsWith('blob:') || tip);
    `),
    120_000,
    "The breed UI analysis did not finish"
  );
  const state = await driver.executeScript(`
    return {
      tip: document.querySelector('.tip span')?.textContent?.trim() || '',
      downloadUrl: document.querySelector(
        'a[download="CBIT-breed-identification.tsv"]'
      )?.href || '',
      visibleRows: document.querySelectorAll('.el-table__body-wrapper tbody tr').length,
      visibleColumns: document.querySelectorAll(
        '.el-table__header-wrapper th:not(.gutter)'
      ).length,
      unrelatedResources: performance.getEntriesByType('resource')
        .map((entry) => entry.name)
        .filter((name) => name.includes('/data/world.js') || name.includes('/images/')),
    };
  `);
  if (state.tip) throw new Error(`Breed UI showed an error: ${state.tip}`);
  if (!state.downloadUrl.startsWith("blob:")) {
    throw new Error("Breed UI did not create the result download");
  }
  if (state.visibleRows !== 6 || state.visibleColumns !== 3) {
    throw new Error(
      `Unexpected breed result shape: ${state.visibleRows} rows x ${state.visibleColumns} columns`
    );
  }
  if (state.unrelatedResources.length) {
    throw new Error(`Breed route loaded home assets: ${state.unrelatedResources.join(", ")}`);
  }
  console.log(
    `Breed UI: pass, ${state.visibleRows} samples analysed locally, result download ready`
  );
} finally {
  await driver.quit();
}
