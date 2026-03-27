import { describe, it, expect } from "vitest";
import { getAllPosts } from "../lib/mdx";
import { BLOG_CORE_TAGS, BLOG_POST_TYPES } from "../lib/mdx";

// Smoke check: verifies that every published blog post has well-formed
// frontmatter before the build or a visitor hits a broken post page.
describe("blog post content integrity", () => {
  const posts = getAllPosts();

  it("has at least one published post", () => {
    expect(posts.length).toBeGreaterThan(0);
  });

  it("has no duplicate slugs", () => {
    const slugs = posts.map((p) => p.slug);
    expect(new Set(slugs).size).toBe(slugs.length);
  });

  it("every post has the required frontmatter fields", () => {
    for (const post of posts) {
      expect(post.frontmatter.title, `${post.slug}: missing title`).toBeTruthy();
      expect(post.frontmatter.description, `${post.slug}: missing description`).toBeTruthy();
      expect(post.frontmatter.date, `${post.slug}: missing date`).toBeTruthy();
      expect(post.frontmatter.type, `${post.slug}: missing type`).toBeTruthy();
      expect(
        BLOG_POST_TYPES.includes(post.frontmatter.type),
        `${post.slug}: invalid type "${post.frontmatter.type}"`
      ).toBe(true);
      expect(Array.isArray(post.frontmatter.tags), `${post.slug}: tags must be an array`).toBe(true);
      expect(post.frontmatter.tags.length, `${post.slug}: tags must not be empty`).toBeGreaterThan(0);
      for (const tag of post.frontmatter.tags) {
        expect(
          BLOG_CORE_TAGS.includes(tag),
          `${post.slug}: invalid core tag "${tag}"`
        ).toBe(true);
      }
      expect(typeof post.frontmatter.published, `${post.slug}: published must be boolean`).toBe("boolean");
    }
  });

  it("every post date is a valid date string", () => {
    for (const post of posts) {
      const d = new Date(post.frontmatter.date);
      expect(isNaN(d.getTime()), `${post.slug}: invalid date "${post.frontmatter.date}"`).toBe(false);
    }
  });

  it("every post has non-empty content", () => {
    for (const post of posts) {
      expect(post.content.trim().length, `${post.slug}: empty content`).toBeGreaterThan(0);
    }
  });

  it("every post has a non-empty reading time", () => {
    for (const post of posts) {
      expect(post.readingTime, `${post.slug}: missing reading time`).toBeTruthy();
    }
  });
});
