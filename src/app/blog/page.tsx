import type { Metadata } from "next";
import { getAllPosts, getAllTags } from "@/lib/mdx";
import { BlogList } from "@/components/sections/blog-list";
import { SectionHeader } from "@/components/ui/section-header";

export const metadata: Metadata = {
  title: "Blog",
  description: "Build notes, deployment lessons, and practical problem solving.",
};

export default function BlogPage() {
  const posts = getAllPosts();
  const tags = getAllTags();

  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <SectionHeader
          eyebrow="Writing"
          title="Notes From the Build"
          description="Build notes, deployment lessons, troubleshooting wins, and practical problem solving from projects in progress."
          className="mb-16"
        />

        {posts.length === 0 ? (
          <div className="text-center py-16">
            <p className="text-muted-foreground">
              Posts coming soon. Stay tuned!
            </p>
          </div>
        ) : (
          <BlogList posts={posts} tags={tags} />
        )}
      </div>
    </div>
  );
}
