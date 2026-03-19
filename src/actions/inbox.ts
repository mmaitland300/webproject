"use server";

import { revalidatePath } from "next/cache";
import { prisma } from "@/lib/prisma";
import { isAdmin } from "@/lib/admin";

async function requireAdmin() {
  const authorized = await isAdmin();
  if (!authorized) throw new Error("Unauthorized");
}

export async function markAsRead(id: string) {
  await requireAdmin();
  await prisma.contactSubmission.update({
    where: { id },
    data: { read: true },
  });
  revalidatePath("/admin/inbox");
}

export async function markAsUnread(id: string) {
  await requireAdmin();
  await prisma.contactSubmission.update({
    where: { id },
    data: { read: false },
  });
  revalidatePath("/admin/inbox");
}

export async function archiveSubmission(id: string) {
  await requireAdmin();
  await prisma.contactSubmission.update({
    where: { id },
    data: { archivedAt: new Date() },
  });
  revalidatePath("/admin/inbox");
}

export async function unarchiveSubmission(id: string) {
  await requireAdmin();
  await prisma.contactSubmission.update({
    where: { id },
    data: { archivedAt: null },
  });
  revalidatePath("/admin/inbox");
}
