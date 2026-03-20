/**
 * Public-facing contact email shown on the site (footer, about, contact page, resume).
 * Set NEXT_PUBLIC_CONTACT_EMAIL in production when a domain alias is ready (e.g. hello@yourdomain.com).
 */
const FALLBACK_PUBLIC_EMAIL = "mmaitland300@gmail.com";

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
