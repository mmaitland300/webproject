import { prisma } from "@/lib/prisma";
import { InboxTabs } from "@/components/sections/inbox-tabs";
import { SectionHeader } from "@/components/ui/section-header";

export const dynamic = "force-dynamic";

export default async function AdminInboxPage() {
  const active = await prisma.contactSubmission.findMany({
    where: { archivedAt: null },
    orderBy: { createdAt: "desc" },
  });

  const archived = await prisma.contactSubmission.findMany({
    where: { archivedAt: { not: null } },
    orderBy: { createdAt: "desc" },
  });

  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <div className="flex items-center justify-between mb-8">
          <div>
            <SectionHeader
              align="left"
              eyebrow="Admin"
              title="Contact Inbox"
              className="space-y-2"
              titleClassName="text-3xl"
            />
            <p className="text-sm text-muted-foreground mt-1">
              {active.length} active &middot; {archived.length} archived
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

        <InboxTabs active={active} archived={archived} />
      </div>
    </div>
  );
}
