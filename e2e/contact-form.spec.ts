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

test("contact form rejects honeypot fill (bot signal)", async ({ page }) => {
  await page.goto("/contact");
  await page.fill('input[name="name"]', "Test");
  await page.fill('input[name="email"]', "test@example.com");
  await page.fill(
    'textarea[name="message"]',
    "This message is long enough for server validation."
  );
  await page.evaluate(() => {
    const hp = document.querySelector(
      'input[name="_hp"]'
    ) as HTMLInputElement | null;
    if (hp) hp.value = "bot-signal";
  });
  await page.locator('button[type="submit"]').click();

  await expect(page.getByText("Please fix the errors below.")).toBeVisible({
    timeout: 10_000,
  });
});

test("contact form valid submit reaches a terminal UI state", async ({
  page,
}) => {
  await page.goto("/contact");
  await page.fill('input[name="name"]', "E2E Test");
  await page.fill('input[name="email"]', "e2e-test@example.com");
  await page.fill(
    'textarea[name="message"]',
    "Playwright trust-path check: valid payload long enough."
  );
  await page.locator('button[type="submit"]').click();

  await expect(
    page.getByText(
      /Message Sent!|Contact form is not configured yet|Failed to send your message|Too many requests/
    )
  ).toBeVisible({ timeout: 20_000 });
});
