import { parseAppEnv } from "@/lib/env";

const LOCAL_SITE_URL = "http://localhost:3000";

function normalizeUrl(value: string) {
  return value.startsWith("http://") || value.startsWith("https://")
    ? value
    : `https://${value}`;
}

export function getSiteUrl() {
  const env = parseAppEnv();
  const configuredUrl = env.NEXT_PUBLIC_SITE_URL;
  if (configuredUrl) {
    return normalizeUrl(configuredUrl);
  }

  const deploymentUrl = env.VERCEL_PROJECT_PRODUCTION_URL ?? env.VERCEL_URL;
  if (deploymentUrl) {
    return normalizeUrl(deploymentUrl);
  }

  return LOCAL_SITE_URL;
}
