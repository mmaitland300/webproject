import { test, expect } from "@playwright/test";

test("unauthenticated access to /admin/inbox redirects to login", async ({
  page,
}) => {
  await page.goto("/admin/inbox");
  await page.waitForURL(/\/admin\/login/, { timeout: 10_000 });
  expect(page.url()).toContain("/admin/login");
});
