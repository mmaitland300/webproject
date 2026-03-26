import { redirect } from "next/navigation";
import { isAdmin } from "@/lib/admin";
import { AdminNav } from "@/components/layout/admin-nav";

export default async function AdminProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const authorized = await isAdmin();
  if (!authorized) {
    redirect("/admin/login");
  }

  return (
    <div className="pt-16">
      <AdminNav />
      <div>{children}</div>
    </div>
  );
}
