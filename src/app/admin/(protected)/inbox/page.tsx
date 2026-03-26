import { prisma } from "@/lib/prisma";
import { requireAdminPage } from "@/lib/admin";
import { InboxTabs } from "@/components/sections/inbox-tabs";
import { SectionHeader } from "@/components/ui/section-header";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 50;

interface Props {
  searchParams: Promise<{ tab?: string; page?: string }>;
}

export default async function AdminInboxPage({ searchParams }: Props) {
  await requireAdminPage();

  const { tab, page: pageParam } = await searchParams;
  const currentTab = tab === "archived" ? "archived" : "active";

  const [activeCount, archivedCount] = await Promise.all([
    prisma.contactSubmission.count({ where: { archivedAt: null } }),
    prisma.contactSubmission.count({ where: { archivedAt: { not: null } } }),
  ]);

  const totalForTab =
    currentTab === "archived" ? archivedCount : activeCount;
  const totalPages = Math.max(1, Math.ceil(totalForTab / PAGE_SIZE));
  const rawPage = parseInt(pageParam ?? "1", 10);
  const pageNum =
    Number.isFinite(rawPage) && rawPage >= 1 ? Math.floor(rawPage) : 1;
  const currentPage = Math.min(pageNum, totalPages);
  const skip = (currentPage - 1) * PAGE_SIZE;

  const submissions = await prisma.contactSubmission.findMany({
    where:
      currentTab === "archived"
        ? { archivedAt: { not: null } }
        : { archivedAt: null },
    orderBy: { createdAt: "desc" },
    skip,
    take: PAGE_SIZE,
  });

  const submissionIds = submissions.map((submission) => submission.id);
  const sentEmails = submissionIds.length
    ? await prisma.sentEmail.findMany({
        where: { submissionId: { in: submissionIds } },
        orderBy: { createdAt: "asc" },
        select: {
          id: true,
          submissionId: true,
          subject: true,
          body: true,
          createdAt: true,
          sentBy: {
            select: {
              name: true,
              email: true,
            },
          },
        },
      })
    : [];

  const sentEmailsBySubmissionId: Record<
    string,
    {
      id: string;
      subject: string;
      body: string;
      createdAt: Date;
      sentByName: string | null;
    }[]
  > = {};

  for (const sentEmail of sentEmails) {
    if (!sentEmail.submissionId) continue;
    if (!sentEmailsBySubmissionId[sentEmail.submissionId]) {
      sentEmailsBySubmissionId[sentEmail.submissionId] = [];
    }
    sentEmailsBySubmissionId[sentEmail.submissionId].push({
      id: sentEmail.id,
      subject: sentEmail.subject,
      body: sentEmail.body,
      createdAt: sentEmail.createdAt,
      sentByName: sentEmail.sentBy?.name ?? sentEmail.sentBy?.email ?? null,
    });
  }

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
          sentEmailsBySubmissionId={sentEmailsBySubmissionId}
          currentTab={currentTab}
          activeCount={activeCount}
          archivedCount={archivedCount}
          currentPage={currentPage}
          totalPages={totalPages}
        />
      </div>
    </div>
  );
}
