"use client";

import { useActionState } from "react";
import { Github } from "lucide-react";
import { Button } from "@/components/ui/button";
import { submitComment, type CommentActionResult } from "@/actions/comments";
import { signInWithGitHub } from "@/actions/auth";

interface CommentFormProps {
  projectSlug: string;
  currentPath: string;
  isSignedIn: boolean;
}

const initial: CommentActionResult = { success: false, message: "" };

export function CommentForm({
  projectSlug,
  currentPath,
  isSignedIn,
}: CommentFormProps) {
  const [state, formAction, isPending] = useActionState(submitComment, initial);

  if (!isSignedIn) {
    return (
      <form
        action={async () => {
          await signInWithGitHub(currentPath);
        }}
      >
        <Button type="submit" variant="outline" size="sm">
          <Github className="mr-2 h-4 w-4" /> Sign in with GitHub to comment
        </Button>
      </form>
    );
  }

  return (
    <form action={formAction} className="space-y-3">
      <input type="hidden" name="projectSlug" value={projectSlug} />
      <textarea
        name="body"
        required
        minLength={3}
        maxLength={2000}
        rows={3}
        placeholder="Leave a comment..."
        className="w-full rounded-lg border border-border bg-card/40 px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-purple-500/40 resize-y"
      />
      {state.message && !state.success && (
        <p className="text-sm text-destructive">{state.message}</p>
      )}
      {state.errors?.body && (
        <p className="text-sm text-destructive">{state.errors.body[0]}</p>
      )}
      {state.success && (
        <p className="text-sm text-emerald-400">{state.message}</p>
      )}
      <Button type="submit" size="sm" disabled={isPending}>
        {isPending ? "Posting..." : "Post comment"}
      </Button>
    </form>
  );
}
