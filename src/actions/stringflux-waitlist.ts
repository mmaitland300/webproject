"use server";

import { z } from "zod/v4";
import { Resend } from "resend";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";
import { headers } from "next/headers";

const waitlistSchema = z.object({
  email: z.string().email("Invalid email address"),
  interest: z.string().max(200).optional(),
  honeypot: z.string().max(0, "Bot detected"),
});

export type WaitlistState = {
  success: boolean;
  message: string;
  errors?: Record<string, string[]>;
};

let ratelimit: Ratelimit | null = null;
function getRatelimit() {
  if (ratelimit) return ratelimit;
  if (
    process.env.UPSTASH_REDIS_REST_URL &&
    process.env.UPSTASH_REDIS_REST_TOKEN
  ) {
    ratelimit = new Ratelimit({
      redis: Redis.fromEnv(),
      limiter: Ratelimit.slidingWindow(3, "60 s"),
      analytics: true,
      prefix: "waitlist",
    });
    return ratelimit;
  }
  return null;
}

export async function joinWaitlist(
  _prevState: WaitlistState,
  formData: FormData
): Promise<WaitlistState> {
  const raw = {
    email: formData.get("email") as string,
    interest: (formData.get("interest") as string) || undefined,
    honeypot: (formData.get("_hp") as string) || "",
  };

  const parsed = waitlistSchema.safeParse(raw);
  if (!parsed.success) {
    const fieldErrors = parsed.error.flatten().fieldErrors;
    const errors: Record<string, string[]> = {};
    for (const [key, msgs] of Object.entries(fieldErrors)) {
      if (msgs) errors[key] = msgs;
    }
    return { success: false, message: "Please fix the errors below.", errors };
  }

  const normalizedEmail = parsed.data.email.toLowerCase().trim();

  // Rate limiting
  const rl = getRatelimit();
  if (rl) {
    const headerStore = await headers();
    const ip =
      headerStore.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";
    const { success: allowed } = await rl.limit(ip);
    if (!allowed) {
      return {
        success: false,
        message: "Too many requests. Please try again in a minute.",
      };
    }
  }

  // Persist signup. upsert is intentional: duplicate email is a no-op, not an error.
  if (process.env.DATABASE_URL) {
    try {
      const { prisma } = await import("@/lib/prisma");
      await prisma.stringFluxWaitlist.upsert({
        where: { email: normalizedEmail },
        update: {},
        create: {
          email: normalizedEmail,
          source: "stringflux-page",
          interest: parsed.data.interest ?? null,
        },
      });
    } catch (err) {
      console.error("StringFlux waitlist persistence failed:", err);
      return {
        success: false,
        message: "Failed to join the waitlist. Please try again later.",
      };
    }
  }

  // Confirmation email - best-effort, does not block success response.
  const resendKey = process.env.RESEND_API_KEY;
  const fromEmail = process.env.CONTACT_FROM_EMAIL;
  if (resendKey && fromEmail) {
    try {
      const resend = new Resend(resendKey);
      await resend.emails.send({
        from: fromEmail,
        to: normalizedEmail,
        subject: "You're on the StringFlux waitlist",
        text: [
          `Hey,`,
          ``,
          `You're on the StringFlux waitlist. I'll reach out when beta access or a release is ready.`,
          ``,
          `You can unsubscribe at any time by replying to this email.`,
          ``,
          `- Matt`,
        ].join("\n"),
      });
    } catch (err) {
      console.error(
        "StringFlux waitlist confirmation email failed (best-effort):",
        err
      );
    }
  }

  return {
    success: true,
    message: "You're on the list. I'll reach out when it's ready.",
  };
}
