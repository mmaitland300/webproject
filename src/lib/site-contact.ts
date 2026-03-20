/**
 * Public-facing contact email shown on the site (footer, about, contact page, resume).
 * Override with NEXT_PUBLIC_CONTACT_EMAIL in Vercel env vars if this ever changes.
 */
const FALLBACK_PUBLIC_EMAIL = "contact@mmaitland.dev";

export function getPublicContactEmail(): string {
  const v = process.env.NEXT_PUBLIC_CONTACT_EMAIL;
  if (typeof v === "string" && v.trim().length > 0) {
    return v.trim();
  }
  return FALLBACK_PUBLIC_EMAIL;
}

export function getPublicMailtoHref(): string {
  return `mailto:${getPublicContactEmail()}`;
}
