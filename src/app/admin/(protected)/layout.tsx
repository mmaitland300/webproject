import Link from "next/link";
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

  return (
    <div>
      <div className="border-b border-border bg-background/60 backdrop-blur-sm">
        <div className="mx-auto flex max-w-4xl gap-4 px-6 py-3 text-sm text-muted-foreground">
          <Link
            href="/admin/inbox"
            className="transition-colors hover:text-foreground"
          >
            Inbox
          </Link>
          <Link
            href="/admin/waitlist"
            className="transition-colors hover:text-foreground"
          >
            Waitlist
          </Link>
          <Link
            href="/admin/comments"
            className="transition-colors hover:text-foreground"
          >
            Comments
          </Link>
        </div>
      </div>
      {children}
    </div>
  );
}
