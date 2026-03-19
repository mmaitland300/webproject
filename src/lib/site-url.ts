const LOCAL_SITE_URL = "http://localhost:3000";

function normalizeUrl(value: string) {
  return value.startsWith("http://") || value.startsWith("https://")
    ? value
    : `https://${value}`;
}

export function getSiteUrl() {
  const configuredUrl = process.env.NEXT_PUBLIC_SITE_URL?.trim();
  if (configuredUrl) {
    return normalizeUrl(configuredUrl);
  }

  const deploymentUrl =
    process.env.VERCEL_PROJECT_PRODUCTION_URL ?? process.env.VERCEL_URL;
  if (deploymentUrl) {
    return normalizeUrl(deploymentUrl.trim());
  }

  return LOCAL_SITE_URL;
}
