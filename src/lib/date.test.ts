import { describe, expect, it } from "vitest";
import { formatDisplayDate, parseDateValue } from "./date";

describe("parseDateValue", () => {
  it("parses date-only strings as UTC midnight", () => {
    expect(parseDateValue("2026-03-18").toISOString()).toBe(
      "2026-03-18T00:00:00.000Z"
    );
  });

  it("returns Date instances unchanged", () => {
    const value = new Date("2026-03-18T15:45:00.000Z");
    expect(parseDateValue(value)).toBe(value);
  });
});

describe("formatDisplayDate", () => {
  it("formats date-only strings without timezone drift", () => {
    expect(formatDisplayDate("2026-03-18")).toBe("March 18, 2026");
  });

  it("supports custom Intl options", () => {
    expect(
      formatDisplayDate("2026-03-18", "en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    ).toBe("Mar 18, 2026");
  });
});
