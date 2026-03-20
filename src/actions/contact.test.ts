import { describe, it, expect } from "vitest";
import { contactSchema } from "./contact";

function parse(data: Record<string, string>) {
  return contactSchema.safeParse(data);
}

const VALID = {
  name: "Jane Doe",
  email: "jane@example.com",
  message: "This is a test message that is long enough.",
  honeypot: "",
};

describe("contact form schema", () => {
  it("accepts a valid submission", () => {
    expect(parse(VALID).success).toBe(true);
  });

  it("rejects an empty name", () => {
    const r = parse({ ...VALID, name: "" });
    expect(r.success).toBe(false);
  });

  it("rejects a name over 100 characters", () => {
    const r = parse({ ...VALID, name: "a".repeat(101) });
    expect(r.success).toBe(false);
  });

  it("rejects an invalid email", () => {
    const r = parse({ ...VALID, email: "not-an-email" });
    expect(r.success).toBe(false);
  });

  it("rejects a message under 10 characters", () => {
    const r = parse({ ...VALID, message: "short" });
    expect(r.success).toBe(false);
  });

  it("rejects a message over 5000 characters", () => {
    const r = parse({ ...VALID, message: "a".repeat(5001) });
    expect(r.success).toBe(false);
  });

  it("rejects a non-empty honeypot (bot detection)", () => {
    const r = parse({ ...VALID, honeypot: "filled by bot" });
    expect(r.success).toBe(false);
  });

  it("accepts an empty honeypot", () => {
    expect(parse({ ...VALID, honeypot: "" }).success).toBe(true);
  });

  it("accepts a message of exactly 10 characters", () => {
    expect(parse({ ...VALID, message: "1234567890" }).success).toBe(true);
  });

  it("accepts a message of exactly 5000 characters", () => {
    expect(parse({ ...VALID, message: "a".repeat(5000) }).success).toBe(true);
  });
});
