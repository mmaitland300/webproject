import { test, expect } from "@playwright/test";

const publicRoutes = [
  "/",
  "/about",
  "/projects",
  "/stringflux",
  "/projects/stringflux",
  "/projects/full-swing-tech-support",
  "/projects/portfolio-site",
  "/blog",
  "/contact",
  "/music",
  "/resume",
];

for (const route of publicRoutes) {
  test(`GET ${route} returns 200 and renders content`, async ({ page }) => {
    const response = await page.goto(route);
    expect(response?.status()).toBe(200);
    await expect(page.locator("main")).toBeVisible();
  });
}

test("admin/login renders without error", async ({ page }) => {
  const response = await page.goto("/admin/login");
  expect(response?.status()).toBe(200);
  const heading = page.locator("h1");
  await expect(heading).toBeVisible();
  const normalized = (await heading.textContent())?.replace(/\s+/g, " ").trim();
  expect(["Admin Login", "Admin Unavailable", "Access Denied"]).toContain(
    normalized
  );
});


