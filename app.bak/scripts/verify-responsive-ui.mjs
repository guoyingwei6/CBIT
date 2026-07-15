import { Builder, Browser } from "selenium-webdriver";
import chrome from "selenium-webdriver/chrome.js";

const argumentsByName = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  argumentsByName.set(process.argv[index], process.argv[index + 1]);
}
const baseUrl = argumentsByName.get("--url") || "http://127.0.0.1:8080";
const routes = ["/home", "/Breed_identification", "/GBC_estimation", "/About"];
const options = new chrome.Options().addArguments(
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--window-size=390,844"
);
options.setMobileEmulation({
  deviceMetrics: { width: 390, height: 844, pixelRatio: 1 },
});
const driver = await new Builder()
  .forBrowser(Browser.CHROME)
  .setChromeOptions(options)
  .build();

try {
  await driver.manage().setTimeouts({ pageLoad: 120_000, script: 120_000 });
  for (const route of routes) {
    await driver.get(`${baseUrl}/#${route}`);
    const state = await driver.executeScript(`
      const root = document.documentElement;
      const width = root.clientWidth;
      return {
        innerWidth,
        clientWidth: width,
        scrollWidth: root.scrollWidth,
        offenders: [...document.body.querySelectorAll('*')]
          .filter((element) => {
            const style = getComputedStyle(element);
            if (style.display === 'none' || style.position === 'fixed') return false;
            const box = element.getBoundingClientRect();
            return box.right > width + 1 || box.left < -1;
          })
          .slice(0, 8)
          .map((element) => ({
            tag: element.tagName,
            className: String(element.className || ''),
            width: Math.round(element.getBoundingClientRect().width),
            right: Math.round(element.getBoundingClientRect().right),
          })),
      };
    `);
    if (state.scrollWidth > state.clientWidth + 1) {
      throw new Error(
        `${route} overflows at ${state.innerWidth}px: ${state.scrollWidth}px document; ` +
          JSON.stringify(state.offenders)
      );
    }
    console.log(`${route}: ${state.clientWidth}px wide, no page overflow`);
  }
} finally {
  await driver.quit();
}
