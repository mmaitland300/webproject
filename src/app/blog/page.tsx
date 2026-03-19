import type { Metadata } from "next";
import { getAllPosts, getAllTags } from "@/lib/mdx";
import { BlogList } from "@/components/sections/blog-list";

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
        <div className="text-center mb-16">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
            <span className="gradient-text">Blog</span>
          </h1>
          <p className="mt-4 text-muted-foreground max-w-lg mx-auto">
            Thoughts on web development, TypeScript, and building things for the
            web.
          </p>
        </div>

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
