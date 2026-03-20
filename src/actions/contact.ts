"use server";

import { Resend } from "resend";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";
import { headers } from "next/headers";
import { contactSchema } from "./contact.contract";

export type ContactState = {
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
      prefix: "contact",
    });
    return ratelimit;
  }
  return null;
}

export async function submitContact(
  _prevState: ContactState,
  formData: FormData
): Promise<ContactState> {
  const raw = {
    name: formData.get("name") as string,
    email: formData.get("email") as string,
    message: formData.get("message") as string,
    honeypot: (formData.get("_hp") as string) || "",
  };

  const parsed = contactSchema.safeParse(raw);
  if (!parsed.success) {
    const fieldErrors = parsed.error.flatten().fieldErrors;
    const errors: Record<string, string[]> = {};
    for (const [key, msgs] of Object.entries(fieldErrors)) {
      if (msgs) errors[key] = msgs;
    }
    return {
      success: false,
      message: "Please fix the errors below.",
      errors,
    };
  }

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

  // Send email via Resend (must succeed before optional DB persistence)
  const resendKey = process.env.RESEND_API_KEY;
  const fromEmail = process.env.CONTACT_FROM_EMAIL;
  const toEmail = process.env.CONTACT_TO_EMAIL;

  if (!resendKey || !fromEmail || !toEmail) {
    console.error("Missing Resend env vars");
    return {
      success: false,
      message: "Contact form is not configured yet. Please try again later.",
    };
  }

  try {
    const resend = new Resend(resendKey);
    await resend.emails.send({
      from: fromEmail,
      to: toEmail,
      replyTo: parsed.data.email,
      subject: `Portfolio Contact: ${parsed.data.name}`,
      text: [
        `Name: ${parsed.data.name}`,
        `Email: ${parsed.data.email}`,
        ``,
        `Message:`,
        parsed.data.message,
      ].join("\n"),
    });
  } catch (err) {
    console.error("Failed to send email:", err);
    return {
      success: false,
      message: "Failed to send your message. Please try again later.",
    };
  }

  // Persist only after email delivery succeeds (avoids orphan inbox rows on misconfig or send failure).
  // Reliability tradeoff: the UI success path is driven by email delivery; admin inbox persistence
  // is best-effort and may be missed if Prisma/DB persistence fails.
  if (process.env.DATABASE_URL) {
    try {
      const { prisma } = await import("@/lib/prisma");
      await prisma.contactSubmission.create({
        data: {
          name: parsed.data.name,
          email: parsed.data.email,
          message: parsed.data.message,
        },
      });
    } catch (err) {
      console.error(
        "Admin inbox persistence failed after successful email delivery (best-effort path):",
        err
      );
    }
  }

  return {
    success: true,
    message: "Thanks for reaching out! I'll get back to you soon.",
  };
}
