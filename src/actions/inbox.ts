"use server";

import { revalidatePath } from "next/cache";
import { prisma } from "@/lib/prisma";
import { isAdmin } from "@/lib/admin";

export type InboxActionResult = {
  success: boolean;
  message: string;
};

async function requireAdmin() {
  const authorized = await isAdmin();
  if (!authorized) {
    return {
      success: false,
      message: "Unauthorized. Please sign in again.",
    } satisfies InboxActionResult;
  }

  return null;
}

async function updateSubmission(
  id: string,
  data: { read?: boolean; archivedAt?: Date | null }
): Promise<InboxActionResult> {
  const unauthorized = await requireAdmin();
  if (unauthorized) {
    return unauthorized;
  }

  try {
    await prisma.contactSubmission.update({
      where: { id },
      data,
    });
    revalidatePath("/admin/inbox");
    return {
      success: true,
      message: "Updated.",
    };
  } catch (error) {
    console.error("Inbox action failed:", error);
    return {
      success: false,
      message:
        "Could not update this message right now. Please try again in a moment.",
    };
  }
}

export async function markAsRead(id: string): Promise<InboxActionResult> {
  return updateSubmission(id, { read: true });
}

export async function markAsUnread(id: string): Promise<InboxActionResult> {
  return updateSubmission(id, { read: false });
}

export async function archiveSubmission(id: string): Promise<InboxActionResult> {
  return updateSubmission(id, { archivedAt: new Date() });
}

export async function unarchiveSubmission(
  id: string
): Promise<InboxActionResult> {
  return updateSubmission(id, { archivedAt: null });
}
