import { describe, it, expect } from "vitest";
import {
  normalizeWaitlistEmail,
  waitlistSchema,
} from "./stringflux-waitlist.contract";

function parse(data: Record<string, string | undefined>) {
  return waitlistSchema.safeParse(data);
}

const VALID = {
  email: "guitarist@example.com",
  honeypot: "",
};

describe("stringflux waitlist schema", () => {
  it("accepts a valid submission with only email", () => {
    expect(parse(VALID).success).toBe(true);
  });

  it("accepts a valid submission with interest filled in", () => {
    expect(
      parse({ ...VALID, interest: "guitar texture layers" }).success
    ).toBe(true);
  });

  it("rejects an invalid email", () => {
    const r = parse({ ...VALID, email: "not-an-email" });
    expect(r.success).toBe(false);
  });

  it("rejects an empty email", () => {
    const r = parse({ ...VALID, email: "" });
    expect(r.success).toBe(false);
  });

  it("rejects a non-empty honeypot (bot detection)", () => {
    const r = parse({ ...VALID, honeypot: "filled by bot" });
    expect(r.success).toBe(false);
  });

  it("accepts an empty honeypot", () => {
    expect(parse({ ...VALID, honeypot: "" }).success).toBe(true);
  });

  it("rejects an interest field over 200 characters", () => {
    const r = parse({ ...VALID, interest: "a".repeat(201) });
    expect(r.success).toBe(false);
  });

  it("accepts an interest field of exactly 200 characters", () => {
    expect(parse({ ...VALID, interest: "a".repeat(200) }).success).toBe(true);
  });

  it("accepts interest as undefined (optional field)", () => {
    const r = parse({ email: VALID.email, honeypot: "", interest: undefined });
    expect(r.success).toBe(true);
  });
});

describe("stringflux waitlist deduplication contract", () => {
  it("two submissions with the same email should produce the same unique key", () => {
    const email = "duplicate@example.com";
    const a = parse({ email, honeypot: "" });
    const b = parse({ email, honeypot: "" });
    expect(a.success).toBe(true);
    expect(b.success).toBe(true);
    if (a.success && b.success) {
      expect(a.data.email).toBe(b.data.email);
    }
  });

  it("normalizes email casing so mixed-case duplicates collapse", () => {
    const a = normalizeWaitlistEmail("TEST@EXAMPLE.COM");
    const b = normalizeWaitlistEmail("test@example.com");
    const c = normalizeWaitlistEmail("Test@Example.COM");
    expect(a).toBe(b);
    expect(b).toBe(c);
    expect(a).toBe("test@example.com");
  });

  it("trims whitespace from email before normalization", () => {
    expect(normalizeWaitlistEmail("  user@test.com  ")).toBe("user@test.com");
  });
});
