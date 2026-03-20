import Link from "next/link";
import { InboxList } from "@/components/sections/inbox-list";
import { cn } from "@/lib/utils";

interface Submission {
  id: string;
  name: string;
  email: string;
  message: string;
  read: boolean;
  archivedAt: Date | null;
  createdAt: Date;
}

interface InboxTabsProps {
  submissions: Submission[];
  currentTab: "active" | "archived";
  activeCount: number;
  archivedCount: number;
  currentPage: number;
  totalPages: number;
}

function inboxHref(tab: "active" | "archived", page: number) {
  const params = new URLSearchParams();
  if (tab === "archived") params.set("tab", "archived");
  if (page > 1) params.set("page", String(page));
  const q = params.toString();
  return `/admin/inbox${q ? `?${q}` : ""}`;
}

export function InboxTabs({
  submissions,
  currentTab,
  activeCount,
  archivedCount,
  currentPage,
  totalPages,
}: InboxTabsProps) {
  return (
    <div>
      <div className="flex gap-1 mb-6 border-b border-border">
        <Link
          href={inboxHref("active", 1)}
          className={cn(
            "px-4 py-2 text-sm font-medium transition-colors -mb-px",
            currentTab === "active"
              ? "border-b-2 border-foreground text-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          Active ({activeCount})
        </Link>
        <Link
          href={inboxHref("archived", 1)}
          className={cn(
            "px-4 py-2 text-sm font-medium transition-colors -mb-px",
            currentTab === "archived"
              ? "border-b-2 border-foreground text-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          Archived ({archivedCount})
        </Link>
      </div>

      {submissions.length === 0 ? (
        <div className="text-center py-16 border border-border rounded-xl bg-card/50">
          <p className="text-muted-foreground">
            {currentTab === "active"
              ? "No messages yet. They\u2019ll appear here when someone uses the contact form."
              : "No archived messages."}
          </p>
        </div>
      ) : (
        <>
          <InboxList submissions={submissions} mode={currentTab} />
          {totalPages > 1 && (
            <div className="mt-8 flex flex-col items-center justify-between gap-3 border-t border-border pt-6 text-sm sm:flex-row">
              {currentPage > 1 ? (
                <Link
                  href={inboxHref(currentTab, currentPage - 1)}
                  className="text-purple-400 hover:text-purple-300"
                >
                  Previous
                </Link>
              ) : (
                <span className="text-muted-foreground">Previous</span>
              )}
              <span className="text-muted-foreground">
                Page {currentPage} of {totalPages}
              </span>
              {currentPage < totalPages ? (
                <Link
                  href={inboxHref(currentTab, currentPage + 1)}
                  className="text-purple-400 hover:text-purple-300"
                >
                  Next
                </Link>
              ) : (
                <span className="text-muted-foreground">Next</span>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
