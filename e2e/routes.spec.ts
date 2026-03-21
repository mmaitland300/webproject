import { test, expect } from "@playwright/test";

const publicRoutes = [
  "/",
  "/about",
  "/projects",
  "/stringflux",
  "/projects/stringflux",
  "/projects/full-swing-tech-support",
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
  const text = await heading.textContent();
  expect(
    text === "Admin Login" || text === "Admin Unavailable"
  ).toBeTruthy();
});
