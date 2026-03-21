import { test, expect } from "@playwright/test";

test("contact form shows server validation on short message", async ({
  page,
}) => {
  await page.goto("/contact");
  await expect(page.locator('button[type="submit"]')).toBeVisible();

  await page.fill('input[name="name"]', "Test");
  await page.fill('input[name="email"]', "test@example.com");
  await page.fill('textarea[name="message"]', "short");
  await page.click('button[type="submit"]');

  await expect(
    page.getByText("Message must be at least 10 characters")
  ).toBeVisible({ timeout: 10_000 });
});

test("contact form renders required fields", async ({ page }) => {
  await page.goto("/contact");
  await expect(page.locator('input[name="name"]')).toBeVisible();
  await expect(page.locator('input[name="email"]')).toBeVisible();
  await expect(page.locator('textarea[name="message"]')).toBeVisible();
  await expect(page.locator('button[type="submit"]')).toBeVisible();
});
