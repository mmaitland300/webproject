import { describe, expect, it } from "vitest";
import {
  getAdminGithubIds,
  getContactDeliveryEnv,
  getResendSenderEnv,
  hasUpstashRedisEnv,
  isAdminAuthEnvConfigured,
  isWaitlistEnvConfigured,
} from "./env";

describe("isWaitlistEnvConfigured", () => {
  it("requires DATABASE_URL", () => {
    expect(isWaitlistEnvConfigured({})).toBe(false);
    expect(isWaitlistEnvConfigured({ DATABASE_URL: "postgres://db" })).toBe(true);
  });
});

describe("isAdminAuthEnvConfigured", () => {
  it("returns false when required fields are missing", () => {
    expect(
      isAdminAuthEnvConfigured({
        DATABASE_URL: "postgres://db",
        AUTH_SECRET: "secret",
      })
    ).toBe(false);
  });

  it("returns true when all required fields are present", () => {
    expect(
      isAdminAuthEnvConfigured({
        DATABASE_URL: "postgres://db",
        AUTH_SECRET: "secret",
        AUTH_GITHUB_ID: "id",
        AUTH_GITHUB_SECRET: "secret",
      })
    ).toBe(true);
  });
});

describe("contact and resend env parsing", () => {
  it("requires full contact delivery env", () => {
    expect(
      getContactDeliveryEnv({
        RESEND_API_KEY: "re_123",
        CONTACT_FROM_EMAIL: "contact@example.com",
      })
    ).toBeNull();
  });

  it("allows resend sender env without CONTACT_TO_EMAIL", () => {
    expect(
      getResendSenderEnv({
        RESEND_API_KEY: "re_123",
        CONTACT_FROM_EMAIL: "contact@example.com",
      })
    ).toEqual({
      RESEND_API_KEY: "re_123",
      CONTACT_FROM_EMAIL: "contact@example.com",
    });
  });
});

describe("misc env helpers", () => {
  it("parses admin github ids", () => {
    expect(getAdminGithubIds({ ADMIN_GITHUB_IDS: "123, 456 , ,789" })).toEqual([
      "123",
      "456",
      "789",
    ]);
  });

  it("detects upstash env presence", () => {
    expect(hasUpstashRedisEnv({})).toBe(false);
    expect(
      hasUpstashRedisEnv({
        UPSTASH_REDIS_REST_URL: "https://example.upstash.io",
        UPSTASH_REDIS_REST_TOKEN: "token",
      })
    ).toBe(true);
  });
});

