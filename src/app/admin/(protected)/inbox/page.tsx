import { prisma } from "@/lib/prisma";
import { InboxTabs } from "@/components/sections/inbox-tabs";
import { SectionHeader } from "@/components/ui/section-header";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 50;

interface Props {
  searchParams: Promise<{ tab?: string }>;
}

export default async function AdminInboxPage({ searchParams }: Props) {
  const { tab } = await searchParams;
  const currentTab = tab === "archived" ? "archived" : "active";

  const [submissions, activeCount, archivedCount] = await Promise.all([
    prisma.contactSubmission.findMany({
      where:
        currentTab === "archived"
          ? { archivedAt: { not: null } }
          : { archivedAt: null },
      orderBy: { createdAt: "desc" },
      take: PAGE_SIZE,
    }),
    prisma.contactSubmission.count({ where: { archivedAt: null } }),
    prisma.contactSubmission.count({ where: { archivedAt: { not: null } } }),
  ]);

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
              {activeCount} active &middot; {archivedCount} archived
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

        <InboxTabs
          submissions={submissions}
          currentTab={currentTab}
          activeCount={activeCount}
          archivedCount={archivedCount}
        />
      </div>
    </div>
  );
}
