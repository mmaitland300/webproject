"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import {
  Mail,
  MailOpen,
  Archive,
  ArchiveRestore,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  markAsRead,
  markAsUnread,
  archiveSubmission,
  unarchiveSubmission,
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
  mode?: "active" | "archived";
}

export function InboxList({ submissions, mode = "active" }: InboxListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  function runInboxAction(
    id: string,
    action: (submissionId: string) => Promise<{ success: boolean; message: string }>
  ) {
    setPendingId(id);
    setFeedback(null);

    startTransition(() => {
      void (async () => {
        try {
          const result = await action(id);
          setFeedback({
            type: result.success ? "success" : "error",
            message: result.message,
          });

          if (result.success) {
            router.refresh();
          }
        } catch {
          setFeedback({
            type: "error",
            message:
              "Could not update this message right now. Please try again in a moment.",
          });
        } finally {
          setPendingId(null);
        }
      })();
    });
  }

  return (
    <div className="space-y-3">
      {feedback && (
        <div
          className={cn(
            "flex items-center gap-2 rounded-lg border p-3 text-sm",
            feedback.type === "error"
              ? "border-destructive/40 bg-destructive/10 text-destructive"
              : "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
          )}
          role="status"
          aria-live="polite"
        >
          {feedback.type === "error" ? (
            <AlertCircle size={16} className="shrink-0" />
          ) : (
            <CheckCircle2 size={16} className="shrink-0" />
          )}
          <span>{feedback.message}</span>
        </div>
      )}

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
                    disabled={isPending && pendingId === sub.id}
                    onClick={() => runInboxAction(sub.id, markAsUnread)}
                  >
                    <Mail size={14} className="mr-1.5" /> Mark unread
                  </Button>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={isPending && pendingId === sub.id}
                    onClick={() => runInboxAction(sub.id, markAsRead)}
                  >
                    <MailOpen size={14} className="mr-1.5" /> Mark read
                  </Button>
                )}
                {mode === "archived" ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={isPending && pendingId === sub.id}
                    onClick={() => runInboxAction(sub.id, unarchiveSubmission)}
                  >
                    <ArchiveRestore size={14} className="mr-1.5" /> Restore
                  </Button>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={isPending && pendingId === sub.id}
                    onClick={() => runInboxAction(sub.id, archiveSubmission)}
                  >
                    <Archive size={14} className="mr-1.5" /> Archive
                  </Button>
                )}
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
