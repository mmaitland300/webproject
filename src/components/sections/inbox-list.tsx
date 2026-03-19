"use client";

import { useState } from "react";
import { Mail, MailOpen, Archive, ArchiveRestore } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  markAsRead,
  markAsUnread,
  archiveSubmission,
} from "@/actions/inbox";
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

interface InboxListProps {
  submissions: Submission[];
}

export function InboxList({ submissions }: InboxListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <div className="space-y-3">
      {submissions.map((sub) => (
        <div
          key={sub.id}
          className={cn(
            "border border-border rounded-xl overflow-hidden transition-colors",
            !sub.read ? "bg-card/80 border-purple-500/20" : "bg-card/30"
          )}
        >
          <button
            onClick={() => setExpandedId(expandedId === sub.id ? null : sub.id)}
            className="w-full text-left p-4 flex items-start gap-3"
          >
            <div className="mt-0.5 shrink-0">
              {sub.read ? (
                <MailOpen size={16} className="text-muted-foreground" />
              ) : (
                <Mail size={16} className="text-purple-400" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <span
                  className={cn(
                    "text-sm font-medium truncate",
                    !sub.read && "text-foreground"
                  )}
                >
                  {sub.name}
                </span>
                <span className="text-xs text-muted-foreground shrink-0">
                  {new Date(sub.createdAt).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">{sub.email}</p>
              <p className="text-sm text-muted-foreground mt-1 line-clamp-1">
                {sub.message}
              </p>
            </div>
          </button>

          {expandedId === sub.id && (
            <div className="px-4 pb-4 border-t border-border pt-3">
              <p className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed mb-4">
                {sub.message}
              </p>
              <div className="flex items-center gap-2">
                {sub.read ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => markAsUnread(sub.id)}
                  >
                    <Mail size={14} className="mr-1.5" /> Mark unread
                  </Button>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => markAsRead(sub.id)}
                  >
                    <MailOpen size={14} className="mr-1.5" /> Mark read
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => archiveSubmission(sub.id)}
                >
                  <Archive size={14} className="mr-1.5" /> Archive
                </Button>
                <a
                  href={`mailto:${sub.email}`}
                  className="text-xs text-purple-400 hover:text-purple-300 ml-auto"
                >
                  Reply via email
                </a>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
