import { test, expect } from "@playwright/test";

test("waitlist form rejects malformed server payload", async ({ page }) => {
  await page.goto("/stringflux");
  const submitButton = page.locator('button[type="submit"]');
  await expect(submitButton).toBeVisible();

  await page.fill('input[name="email"]', "tester@example.com");
  await page.evaluate(() => {
    const hp = document.querySelector('input[name="_hp"]') as HTMLInputElement | null;
    if (hp) hp.value = "bot-signal";
  });
  await submitButton.click();

  await expect(page.getByText("Please fix the errors below.")).toBeVisible({
    timeout: 10_000,
  });
  await expect(submitButton).toBeVisible();
});
