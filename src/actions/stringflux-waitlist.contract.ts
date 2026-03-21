import { z } from "zod/v4";

export const waitlistSchema = z.object({
  email: z.string().email("Invalid email address"),
  interest: z.string().trim().max(200).optional(),
  honeypot: z.string().max(0, "Bot detected"),
});

export function normalizeWaitlistEmail(email: string): string {
  return email.toLowerCase().trim();
}
