import { MessageSquare } from "lucide-react";
import { prisma } from "@/lib/prisma";
import { parseAppEnv } from "@/lib/env";
import { isAdminAuthConfigured } from "@/lib/feature-config";
import { getSessionUser } from "@/lib/session";
import { isAdmin } from "@/lib/admin";
import { CommentForm } from "@/components/sections/comment-form";
import { CommentList, type CommentData } from "@/components/sections/comment-list";

interface ProjectCommentsProps {
  projectSlug: string;
  currentPath: string;
}

export async function ProjectComments({
  projectSlug,
  currentPath,
}: ProjectCommentsProps) {
  if (!parseAppEnv().DATABASE_URL) return null;

  const authConfigured = isAdminAuthConfigured();
  const [user, admin] = authConfigured
    ? await Promise.all([getSessionUser(), isAdmin()])
    : [null, false];

  const where = admin
    ? { projectSlug }
    : { projectSlug, hidden: false };

  const rawComments = await prisma.projectComment.findMany({
    where,
    orderBy: { createdAt: "asc" },
    select: {
      id: true,
      body: true,
      hidden: true,
      createdAt: true,
      user: { select: { name: true, image: true } },
    },
  });

  const comments: CommentData[] = rawComments.map((c) => ({
    ...c,
    createdAt: c.createdAt,
  }));

  return (
    <section className="mt-16 rounded-xl border border-border bg-card/40 p-6">
      <div className="mb-4 flex items-center gap-2">
        <MessageSquare className="h-5 w-5 text-purple-400" />
        <h2 className="text-xl font-semibold">Comments</h2>
        {comments.length > 0 && (
          <span className="text-sm text-muted-foreground">
            ({comments.length})
          </span>
        )}
      </div>

      <CommentList comments={comments} isAdmin={admin} />

      {authConfigured && (
        <div className="mt-6 border-t border-border pt-4">
          <CommentForm
            projectSlug={projectSlug}
            currentPath={currentPath}
            isSignedIn={!!user}
          />
        </div>
      )}
    </section>
  );
}
