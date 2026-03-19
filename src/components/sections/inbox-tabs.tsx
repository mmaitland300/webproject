"use client";

import { useState } from "react";
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
  active: Submission[];
  archived: Submission[];
}

export function InboxTabs({ active, archived }: InboxTabsProps) {
  const [tab, setTab] = useState<"active" | "archived">("active");
  const items = tab === "active" ? active : archived;

  return (
    <div>
      <div className="flex gap-1 mb-6 border-b border-border">
        <button
          onClick={() => setTab("active")}
          className={cn(
            "px-4 py-2 text-sm font-medium transition-colors -mb-px",
            tab === "active"
              ? "border-b-2 border-foreground text-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          Active ({active.length})
        </button>
        <button
          onClick={() => setTab("archived")}
          className={cn(
            "px-4 py-2 text-sm font-medium transition-colors -mb-px",
            tab === "archived"
              ? "border-b-2 border-foreground text-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          Archived ({archived.length})
        </button>
      </div>

      {items.length === 0 ? (
        <div className="text-center py-16 border border-border rounded-xl bg-card/50">
          <p className="text-muted-foreground">
            {tab === "active"
              ? "No messages yet. They\u2019ll appear here when someone uses the contact form."
              : "No archived messages."}
          </p>
        </div>
      ) : (
        <InboxList submissions={items} mode={tab} />
      )}
    </div>
  );
}
