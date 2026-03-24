import { auth } from "@/lib/auth";
import { isAdminAuthConfigured } from "@/lib/feature-config";

export type SessionUser = {
  id: string;
  name: string | null;
  image: string | null;
};

export async function getSessionUser(): Promise<SessionUser | null> {
  if (!isAdminAuthConfigured()) return null;

  const session = await auth();
  if (!session?.user?.id) return null;

  return {
    id: session.user.id,
    name: session.user.name ?? null,
    image: session.user.image ?? null,
  };
}
