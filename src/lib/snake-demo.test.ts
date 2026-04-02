import { describe, expect, it, afterEach } from "vitest";
import { getSnakeDemoUrl } from "./snake-demo";

describe("getSnakeDemoUrl", () => {
  afterEach(() => {
    delete process.env.NEXT_PUBLIC_SNAKE_DEMO_URL;
  });

  it("returns undefined when unset or empty", () => {
    delete process.env.NEXT_PUBLIC_SNAKE_DEMO_URL;
    expect(getSnakeDemoUrl()).toBeUndefined();
    process.env.NEXT_PUBLIC_SNAKE_DEMO_URL = "   ";
    expect(getSnakeDemoUrl()).toBeUndefined();
  });

  it("returns the URL when valid http(s)", () => {
    process.env.NEXT_PUBLIC_SNAKE_DEMO_URL = "https://demo.example/app";
    expect(getSnakeDemoUrl()).toBe("https://demo.example/app");
  });

  it("rejects non-http schemes", () => {
    process.env.NEXT_PUBLIC_SNAKE_DEMO_URL = "ftp://bad";
    expect(getSnakeDemoUrl()).toBeUndefined();
  });
});
