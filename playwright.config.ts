import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: "http://localhost:3000",
    headless: true,
  },
  projects: [
    { name: "chromium", use: { browserName: "chromium" } },
  ],
  webServer: {
    command: "npm start",
    url: "http://localhost:3000",
    env: {
      AUTH_TRUST_HOST: "true",
    },
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
