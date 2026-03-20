import { describe, expect, it } from "vitest";
import {
  isAdminAuthConfigured,
  isWaitlistConfigured,
} from "./feature-config";

describe("isWaitlistConfigured", () => {
  it("requires a database URL", () => {
    expect(isWaitlistConfigured({})).toBe(false);
    expect(isWaitlistConfigured({ DATABASE_URL: "postgres://db" })).toBe(true);
  });
});

describe("isAdminAuthConfigured", () => {
  it("returns false when any required admin auth env var is missing", () => {
    expect(
      isAdminAuthConfigured({
        DATABASE_URL: "postgres://db",
        AUTH_SECRET: "secret",
        AUTH_GITHUB_ID: "github-id",
      })
    ).toBe(false);
  });

  it("returns true when database and GitHub auth env vars are all present", () => {
    expect(
      isAdminAuthConfigured({
        DATABASE_URL: "postgres://db",
        AUTH_SECRET: "secret",
        AUTH_GITHUB_ID: "github-id",
        AUTH_GITHUB_SECRET: "github-secret",
      })
    ).toBe(true);
  });
});
