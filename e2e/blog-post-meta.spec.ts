import { test, expect } from "@playwright/test";

/** Stable published post; blog generateMetadata sets article OG fields. */
const BLOG_POST_PATH = "/blog/building-this-site";

test("blog post exposes article Open Graph metadata", async ({ page }) => {
  await page.goto(BLOG_POST_PATH);
  expect(page.url()).toContain(BLOG_POST_PATH);

  await expect(
    page.locator('meta[property="og:type"]')
  ).toHaveAttribute("content", "article");

  const published = page.locator('meta[property="article:published_time"]');
  await expect(published).toHaveCount(1);
  const content = await published.getAttribute("content");
  expect(content).toBeTruthy();
  expect(content?.length).toBeGreaterThan(4);
});
