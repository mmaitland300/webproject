import { z } from "zod/v4";

export const commentSchema = z.object({
  projectSlug: z.string().min(1).max(100),
  body: z
    .string()
    .trim()
    .min(3, "Comment must be at least 3 characters")
    .max(2000, "Comment must be under 2000 characters"),
});
