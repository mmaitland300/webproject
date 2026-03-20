import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { getSiteUrl } from "./site-url";

const ORIGINAL_ENV = { ...process.env };

beforeEach(() => {
  delete process.env.NEXT_PUBLIC_SITE_URL;
  delete process.env.VERCEL_PROJECT_PRODUCTION_URL;
  delete process.env.VERCEL_URL;
});

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
});

describe("getSiteUrl", () => {
  it("returns NEXT_PUBLIC_SITE_URL when set", () => {
    process.env.NEXT_PUBLIC_SITE_URL = "https://mmaitland.dev";
    expect(getSiteUrl()).toBe("https://mmaitland.dev");
  });

  it("prepends https:// when NEXT_PUBLIC_SITE_URL has no protocol", () => {
    process.env.NEXT_PUBLIC_SITE_URL = "mmaitland.dev";
    expect(getSiteUrl()).toBe("https://mmaitland.dev");
  });

  it("prefers NEXT_PUBLIC_SITE_URL over VERCEL_PROJECT_PRODUCTION_URL", () => {
    process.env.NEXT_PUBLIC_SITE_URL = "https://mmaitland.dev";
    process.env.VERCEL_PROJECT_PRODUCTION_URL = "https://other.vercel.app";
    expect(getSiteUrl()).toBe("https://mmaitland.dev");
  });

  it("falls back to VERCEL_PROJECT_PRODUCTION_URL when NEXT_PUBLIC_SITE_URL is unset", () => {
    process.env.VERCEL_PROJECT_PRODUCTION_URL = "myproject.vercel.app";
    expect(getSiteUrl()).toBe("https://myproject.vercel.app");
  });

  it("falls back to VERCEL_URL when production URL is unset", () => {
    process.env.VERCEL_URL = "myproject-git-main.vercel.app";
    expect(getSiteUrl()).toBe("https://myproject-git-main.vercel.app");
  });

  it("falls back to localhost when no env vars are set", () => {
    expect(getSiteUrl()).toBe("http://localhost:3000");
  });

  it("trims whitespace from the configured URL", () => {
    process.env.NEXT_PUBLIC_SITE_URL = "  https://mmaitland.dev  ";
    expect(getSiteUrl()).toBe("https://mmaitland.dev");
  });
});
