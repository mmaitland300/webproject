import type { Metadata } from "next";
import { getAllPosts, getAllTags } from "@/lib/mdx";
import { BlogList } from "@/components/sections/blog-list";
import { SectionHeader } from "@/components/ui/section-header";

export const metadata: Metadata = {
  title: "Blog",
  description: "Thoughts on web development, TypeScript, and building things.",
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
          description="Thoughts on web development, TypeScript, problem solving, and the process of building things that hold up."
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
