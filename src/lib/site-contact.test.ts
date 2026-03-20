import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { getPublicContactEmail, getPublicMailtoHref } from "./site-contact";

const ORIGINAL_ENV = { ...process.env };

beforeEach(() => {
  delete process.env.NEXT_PUBLIC_CONTACT_EMAIL;
});

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
});

describe("getPublicContactEmail", () => {
  it("returns the branded fallback when env var is unset", () => {
    expect(getPublicContactEmail()).toBe("contact@mmaitland.dev");
  });

  it("returns NEXT_PUBLIC_CONTACT_EMAIL when set", () => {
    process.env.NEXT_PUBLIC_CONTACT_EMAIL = "hello@mmaitland.dev";
    expect(getPublicContactEmail()).toBe("hello@mmaitland.dev");
  });

  it("trims whitespace from the env var", () => {
    process.env.NEXT_PUBLIC_CONTACT_EMAIL = "  hello@mmaitland.dev  ";
    expect(getPublicContactEmail()).toBe("hello@mmaitland.dev");
  });

  it("ignores an empty-string env var and returns the fallback", () => {
    process.env.NEXT_PUBLIC_CONTACT_EMAIL = "   ";
    expect(getPublicContactEmail()).toBe("contact@mmaitland.dev");
  });
});

describe("getPublicMailtoHref", () => {
  it("returns a mailto: href using the resolved email", () => {
    expect(getPublicMailtoHref()).toBe("mailto:contact@mmaitland.dev");
  });

  it("reflects NEXT_PUBLIC_CONTACT_EMAIL when set", () => {
    process.env.NEXT_PUBLIC_CONTACT_EMAIL = "hello@mmaitland.dev";
    expect(getPublicMailtoHref()).toBe("mailto:hello@mmaitland.dev");
  });
});
