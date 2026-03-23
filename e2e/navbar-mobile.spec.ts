import { test, expect } from "@playwright/test";

test.describe("mobile navbar", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("opens menu, shows nav links, locks scroll, closes on Escape", async ({
    page,
  }) => {
    await page.goto("/");

    const toggle = page.getByRole("button", { name: "Open menu" });
    await expect(toggle).toBeVisible();
    await expect(toggle).toHaveAttribute("aria-expanded", "false");
    await toggle.click();

    const closeToggle = page.getByRole("button", { name: "Close menu" });
    await expect(closeToggle).toHaveAttribute("aria-expanded", "true");

    const menu = page.getByTestId("mobile-nav-menu");
    await expect(menu).toBeVisible();
    await expect(menu.getByRole("link", { name: "Projects" })).toBeVisible();

    const overflowWhileOpen = await page.evaluate(
      () => window.getComputedStyle(document.body).overflow
    );
    expect(overflowWhileOpen).toBe("hidden");

    await page.keyboard.press("Escape");
    await expect(menu).not.toBeVisible();
    const reopenToggle = page.getByRole("button", { name: "Open menu" });
    await expect(reopenToggle).toBeVisible();
    await expect(reopenToggle).toHaveAttribute("aria-expanded", "false");
    const overflowAfterClose = await page.evaluate(
      () => window.getComputedStyle(document.body).overflow
    );
    expect(overflowAfterClose).not.toBe("hidden");
  });

  test("closes menu on outside click", async ({ page }) => {
    await page.goto("/");

    await page.getByRole("button", { name: "Open menu" }).click();
    const menu = page.getByTestId("mobile-nav-menu");
    await expect(menu).toBeVisible();

    await page.locator("main").click({ position: { x: 20, y: 400 } });
    await expect(menu).not.toBeVisible();
    await expect(
      page.getByRole("button", { name: "Open menu" })
    ).toHaveAttribute("aria-expanded", "false");
    const overflowAfterClose = await page.evaluate(
      () => window.getComputedStyle(document.body).overflow
    );
    expect(overflowAfterClose).not.toBe("hidden");
  });
});
