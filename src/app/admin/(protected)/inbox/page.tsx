import { prisma } from "@/lib/prisma";
import { InboxList } from "@/components/sections/inbox-list";

export const dynamic = "force-dynamic";

export default async function AdminInboxPage() {
  const submissions = await prisma.contactSubmission.findMany({
    where: { archivedAt: null },
    orderBy: { createdAt: "desc" },
  });

  const archivedCount = await prisma.contactSubmission.count({
    where: { archivedAt: { not: null } },
  });

  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">
              Contact <span className="gradient-text">Inbox</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              {submissions.length} active &middot; {archivedCount} archived
            </p>
          </div>
          <form
            action={async () => {
              "use server";
              const { signOut } = await import("@/lib/auth");
              await signOut({ redirectTo: "/" });
            }}
          >
            <button
              type="submit"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Sign out
            </button>
          </form>
        </div>

        {submissions.length === 0 ? (
          <div className="text-center py-16 border border-border rounded-xl bg-card/50">
            <p className="text-muted-foreground">
              No messages yet. They&apos;ll appear here when someone uses the
              contact form.
            </p>
          </div>
        ) : (
          <InboxList submissions={submissions} />
        )}
      </div>
    </div>
  );
}
