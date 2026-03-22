import { parseAppEnv } from "@/lib/env";

/**
 * Public-facing contact email shown on the site (footer, about, contact page, resume).
 * Override with NEXT_PUBLIC_CONTACT_EMAIL in Vercel env vars if this ever changes.
 */
const FALLBACK_PUBLIC_EMAIL = "contact@mmaitland.dev";

export function getPublicContactEmail(): string {
  const publicContactEmail = parseAppEnv().NEXT_PUBLIC_CONTACT_EMAIL;
  if (publicContactEmail) {
    return publicContactEmail;
  }
  return FALLBACK_PUBLIC_EMAIL;
}

export function getPublicMailtoHref(): string {
  return `mailto:${getPublicContactEmail()}`;
}
