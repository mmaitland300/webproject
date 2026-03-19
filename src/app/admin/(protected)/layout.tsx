import { redirect } from "next/navigation";
import { isAdmin } from "@/lib/admin";

export default async function AdminProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const authorized = await isAdmin();
  if (!authorized) {
    redirect("/admin/login");
  }

  return <>{children}</>;
}
