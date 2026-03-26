import { z } from "zod/v4";

const cuidLikeId = z
  .string()
  .trim()
  .regex(/^c[a-z0-9]{20,}$/i, "Invalid submission id.");

const subjectSchema = z
  .string()
  .trim()
  .min(1, "Subject is required.")
  .max(200, "Subject must be 200 characters or less.");

const bodySchema = z
  .string()
  .trim()
  .min(1, "Message body is required.")
  .max(50000, "Message body is too long.");

export const replyToSubmissionSchema = z.object({
  submissionId: cuidLikeId,
  subject: subjectSchema,
  body: bodySchema,
});

export const sendComposeEmailSchema = z.object({
  toEmail: z.string().trim().email("Recipient email is invalid."),
  toName: z
    .string()
    .trim()
    .max(120, "Recipient name must be 120 characters or less.")
    .optional(),
  subject: subjectSchema,
  body: bodySchema,
});

export type ReplyToSubmissionInput = z.infer<typeof replyToSubmissionSchema>;
export type SendComposeEmailInput = z.infer<typeof sendComposeEmailSchema>;
