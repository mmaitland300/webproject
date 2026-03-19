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
}

export function InboxTabs({
  submissions,
  currentTab,
  activeCount,
  archivedCount,
}: InboxTabsProps) {
  return (
    <div>
      <div className="flex gap-1 mb-6 border-b border-border">
        <Link
          href="/admin/inbox"
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
          href="/admin/inbox?tab=archived"
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
        <InboxList submissions={submissions} mode={currentTab} />
      )}
    </div>
  );
}
