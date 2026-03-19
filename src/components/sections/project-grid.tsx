"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Beaker } from "lucide-react";
import { ProjectCard } from "@/components/sections/project-card";
import { Separator } from "@/components/ui/separator";
import { getExperiments, getFeaturedProjects } from "@/content/projects";
import { cn } from "@/lib/utils";

export function ProjectGrid() {
  const featured = getFeaturedProjects();
  const experiments = getExperiments();

  const allTags = Array.from(new Set(featured.flatMap((p) => p.tags))).sort();
  const [activeTag, setActiveTag] = useState<string | null>(null);

  const filtered = activeTag
    ? featured.filter((p) => p.tags.includes(activeTag))
    : featured;

  return (
    <div>
      <div className="mb-10 flex flex-wrap justify-center gap-2">
        <button
          onClick={() => setActiveTag(null)}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            !activeTag
              ? "border border-purple-500/30 bg-purple-500/20 text-purple-400"
              : "bg-muted text-muted-foreground hover:text-foreground"
          )}
        >
          All
        </button>
        {allTags.map((tag) => (
          <button
            key={tag}
            onClick={() => setActiveTag(tag === activeTag ? null : tag)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              tag === activeTag
                ? "border border-purple-500/30 bg-purple-500/20 text-purple-400"
                : "bg-muted text-muted-foreground hover:text-foreground"
            )}
          >
            {tag}
          </button>
        ))}
      </div>

      <p className="mb-8 text-center text-xs text-muted-foreground">
        Some GitHub links point to private or temporarily unavailable
        repositories while screenshots, READMEs, and case studies are being
        cleaned up. Source access is available on request.
      </p>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {filtered.map((project, i) => (
          <ProjectCard key={project.slug} project={project} index={i} />
        ))}
      </div>

      {filtered.length === 0 && (
        <p className="py-12 text-center text-muted-foreground">
          No projects match that filter.
        </p>
      )}

      {experiments.length > 0 && (
        <>
          <Separator className="my-16 bg-border" />

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mb-10 text-center"
          >
            <div className="mb-3 flex items-center justify-center gap-2">
              <Beaker className="h-5 w-5 text-cyan-400" />
              <h2 className="text-2xl font-bold tracking-tight">Experiments</h2>
            </div>
            <p className="mx-auto max-w-md text-sm text-muted-foreground">
              Side projects and interactive toys, small-scope explorations built
              for fun and learning.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
            {experiments.map((project, i) => (
              <ProjectCard
                key={project.slug}
                project={project}
                index={i}
                compact
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
