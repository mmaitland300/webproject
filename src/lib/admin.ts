import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";
import { isAdminAuthConfigured } from "@/lib/feature-config";
import { prisma } from "@/lib/prisma";

export async function isAdmin(): Promise<boolean> {
  if (!isAdminAuthConfigured()) return false;

  const session = await auth();
  if (!session?.user?.id) return false;

  const adminIds = process.env.ADMIN_GITHUB_IDS?.split(",").map((s) =>
    s.trim()
  );
  if (!adminIds?.length) return false;

  const account = await prisma.account.findFirst({
    where: {
      userId: session.user.id,
      provider: "github",
    },
    select: { providerAccountId: true },
  });

  if (!account) return false;
  return adminIds.includes(account.providerAccountId);
}

export async function requireAdminPage(): Promise<void> {
  if (await isAdmin()) {
    return;
  }

  redirect("/admin/login");
}
