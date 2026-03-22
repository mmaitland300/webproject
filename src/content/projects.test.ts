import { describe, it, expect } from "vitest";
import {
  projects,
  getFeaturedProjects,
  getHomepageFeaturedProjects,
  getExperiments,
} from "./projects";

describe("projects data integrity", () => {
  it("has no duplicate slugs", () => {
    const slugs = projects.map((p) => p.slug);
    const unique = new Set(slugs);
    expect(unique.size).toBe(slugs.length);
  });

  it("every project has required fields", () => {
    for (const p of projects) {
      expect(p.slug, `${p.slug} missing slug`).toBeTruthy();
      expect(p.title, `${p.slug} missing title`).toBeTruthy();
      expect(p.description, `${p.slug} missing description`).toBeTruthy();
      expect(p.tags, `${p.slug} missing tags`).toBeDefined();
      expect(p.tags.length, `${p.slug} has empty tags`).toBeGreaterThan(0);
      expect(["featured", "experiment"], `${p.slug} invalid category`).toContain(
        p.category
      );
    }
  });

  it("caseStudy links follow /projects/<slug> pattern", () => {
    for (const p of projects) {
      if (p.caseStudy) {
        expect(p.caseStudy).toMatch(/^\/projects\/.+/);
      }
    }
  });

  it("getFeaturedProjects returns only featured category", () => {
    const featured = getFeaturedProjects();
    expect(featured.length).toBeGreaterThan(0);
    for (const p of featured) {
      expect(p.category).toBe("featured");
    }
  });

  it("getExperiments returns only experiment category", () => {
    const experiments = getExperiments();
    expect(experiments.length).toBeGreaterThan(0);
    for (const p of experiments) {
      expect(p.category).toBe("experiment");
    }
  });

  it("getHomepageFeaturedProjects resolves without throwing", () => {
    expect(() => getHomepageFeaturedProjects()).not.toThrow();
  });

  it("homepage featured projects are all valid project entries", () => {
    const allSlugs = new Set(projects.map((p) => p.slug));
    for (const p of getHomepageFeaturedProjects()) {
      expect(allSlugs.has(p.slug)).toBe(true);
    }
  });
});

