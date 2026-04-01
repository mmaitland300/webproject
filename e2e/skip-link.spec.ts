import { test, expect } from "@playwright/test";

test.describe("skip link", () => {
  test("first Tab focuses skip link; activating it moves focus to main content", async ({
    page,
  }) => {
    await page.goto("/");

    await page.keyboard.press("Tab");

    const skip = page.getByRole("link", { name: "Skip to content" });
    await expect(skip).toBeFocused();
    await expect(skip).toBeVisible();

    await skip.press("Enter");

    const target = page.locator("#main-content");
    await expect(target).toBeFocused();
    await expect
      .poll(() => page.evaluate(() => window.scrollY))
      .toBeGreaterThan(80);
  });
});
