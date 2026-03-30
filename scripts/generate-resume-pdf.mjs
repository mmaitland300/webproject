/**
 * Renders /resume with Chromium and writes public/resume.pdf.
 * Requires a running site (dev or production server) on the origin below.
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const outPath = join(root, "public", "resume.pdf");

const baseURL = (process.env.RESUME_PDF_ORIGIN ?? "http://localhost:3000").replace(
  /\/$/,
  ""
);

async function main() {
  await mkdir(dirname(outPath), { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  try {
    await page.goto(`${baseURL}/resume`, {
      waitUntil: "load",
      timeout: 60_000,
    });
    await page.pdf({
      path: outPath,
      format: "Letter",
      printBackground: true,
      margin: {
        top: "0.45in",
        right: "0.45in",
        bottom: "0.45in",
        left: "0.45in",
      },
    });
  } finally {
    await browser.close();
  }

  console.log(`Wrote ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
