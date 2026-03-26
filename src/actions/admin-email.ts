"use server";

import { revalidatePath } from "next/cache";
import { Resend } from "resend";
import { auth } from "@/lib/auth";
import { isAdmin } from "@/lib/admin";
import { getContactDeliveryEnv } from "@/lib/env";
import { prisma } from "@/lib/prisma";
import { normalizeReplySubject } from "@/lib/email-subject";
import {
  replyToSubmissionSchema,
  sendComposeEmailSchema,
  type ReplyToSubmissionInput,
  type SendComposeEmailInput,
} from "./admin-email.contract";

type FieldErrors = Record<string, string[]>;

export type AdminEmailActionResult = {
  success: boolean;
  message: string;
  warning?: string;
  historySaved: boolean;
  markedRead: boolean;
  errors?: FieldErrors;
};

function flattenFieldErrors(error: {
  flatten: () => { fieldErrors: Record<string, string[] | undefined> };
}): FieldErrors {
  const fieldErrors = error.flatten().fieldErrors;
  const errors: FieldErrors = {};
  for (const [key, value] of Object.entries(fieldErrors)) {
    if (value?.length) {
      errors[key] = value;
    }
  }
  return errors;
}

function failedResult(
  message: string,
  extras: Partial<AdminEmailActionResult> = {}
): AdminEmailActionResult {
  return {
    success: false,
    message,
    historySaved: false,
    markedRead: false,
    ...extras,
  };
}

async function requireAdminUserId(): Promise<string | AdminEmailActionResult> {
  const authorized = await isAdmin();
  if (!authorized) {
    return failedResult("Unauthorized. Please sign in again.");
  }

  const session = await auth();
  if (!session?.user?.id) {
    return failedResult("Unauthorized. Please sign in again.");
  }

  return session.user.id;
}

function getResendMessageId(result: unknown): string | null {
  if (!result || typeof result !== "object") return null;
  if (!("data" in result)) return null;

  const data = (result as { data?: unknown }).data;
  if (!data || typeof data !== "object") return null;
  if (!("id" in data)) return null;

  const id = (data as { id?: unknown }).id;
  return typeof id === "string" && id.length > 0 ? id : null;
}

export async function replyToSubmission(
  input: ReplyToSubmissionInput
): Promise<AdminEmailActionResult> {
  const userId = await requireAdminUserId();
  if (typeof userId !== "string") {
    return userId;
  }

  const parsed = replyToSubmissionSchema.safeParse(input);
  if (!parsed.success) {
    return failedResult("Please fix the errors below.", {
      errors: flattenFieldErrors(parsed.error),
    });
  }

  const env = getContactDeliveryEnv();
  if (!env) {
    console.error("Missing Resend env vars");
    return failedResult("Email delivery is not configured.");
  }

  const submission = await prisma.contactSubmission.findUnique({
    where: { id: parsed.data.submissionId },
    select: { id: true, email: true },
  });

  if (!submission) {
    return failedResult("That message no longer exists.");
  }

  const subject = normalizeReplySubject(parsed.data.subject);

  try {
    const resend = new Resend(env.RESEND_API_KEY);
    const sendResult = await resend.emails.send({
      from: env.CONTACT_FROM_EMAIL,
      to: submission.email,
      replyTo: env.CONTACT_TO_EMAIL,
      subject,
      text: parsed.data.body,
    });

    const resendMessageId = getResendMessageId(sendResult);
    let historySaved = false;
    let markedRead = false;
    const warnings: string[] = [];

    try {
      await prisma.sentEmail.create({
        data: {
          submissionId: submission.id,
          sentByUserId: userId,
          toEmail: submission.email,
          subject,
          body: parsed.data.body,
          fromEmail: env.CONTACT_FROM_EMAIL,
          replyToEmail: env.CONTACT_TO_EMAIL,
          resendMessageId: resendMessageId ?? undefined,
        },
      });
      historySaved = true;
    } catch (error) {
      console.error("Reply delivered but SentEmail save failed:", error);
      warnings.push("Sent history could not be saved.");
    }

    try {
      await prisma.contactSubmission.update({
        where: { id: submission.id },
        data: { read: true },
      });
      markedRead = true;
    } catch (error) {
      console.error("Reply delivered but read flag update failed:", error);
      warnings.push("The message could not be marked as read.");
    }

    revalidatePath("/admin/inbox");
    revalidatePath("/admin/sent");

    return {
      success: true,
      message:
        warnings.length > 0
          ? "Reply sent with warnings."
          : "Reply sent successfully.",
      warning: warnings.length > 0 ? warnings.join(" ") : undefined,
      historySaved,
      markedRead,
    };
  } catch (error) {
    console.error("Failed to send reply email:", error);
    return failedResult("Could not send the reply. Please try again.");
  }
}

export async function sendComposeEmail(
  input: SendComposeEmailInput
): Promise<AdminEmailActionResult> {
  const userId = await requireAdminUserId();
  if (typeof userId !== "string") {
    return userId;
  }

  const parsed = sendComposeEmailSchema.safeParse(input);
  if (!parsed.success) {
    return failedResult("Please fix the errors below.", {
      errors: flattenFieldErrors(parsed.error),
    });
  }

  const env = getContactDeliveryEnv();
  if (!env) {
    console.error("Missing Resend env vars");
    return failedResult("Email delivery is not configured.");
  }

  try {
    const resend = new Resend(env.RESEND_API_KEY);
    const sendResult = await resend.emails.send({
      from: env.CONTACT_FROM_EMAIL,
      to: parsed.data.toEmail,
      replyTo: env.CONTACT_TO_EMAIL,
      subject: parsed.data.subject,
      text: parsed.data.body,
    });

    const resendMessageId = getResendMessageId(sendResult);
    let historySaved = false;

    try {
      await prisma.sentEmail.create({
        data: {
          sentByUserId: userId,
          toEmail: parsed.data.toEmail,
          toName: parsed.data.toName,
          subject: parsed.data.subject,
          body: parsed.data.body,
          fromEmail: env.CONTACT_FROM_EMAIL,
          replyToEmail: env.CONTACT_TO_EMAIL,
          resendMessageId: resendMessageId ?? undefined,
        },
      });
      historySaved = true;
    } catch (error) {
      console.error("Compose delivered but SentEmail save failed:", error);
    }

    revalidatePath("/admin/sent");
    revalidatePath("/admin/inbox");

    return {
      success: true,
      message: historySaved
        ? "Email sent successfully."
        : "Email sent, but Sent history could not be saved.",
      warning: historySaved
        ? undefined
        : "Sent history could not be saved. Check logs for details.",
      historySaved,
      markedRead: false,
    };
  } catch (error) {
    console.error("Failed to send compose email:", error);
    return failedResult("Could not send the email. Please try again.");
  }
}
