"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Separator } from "@/components/ui/separator";
import { Beaker } from "lucide-react";
import { ProjectCard } from "@/components/sections/project-card";
import { getFeaturedProjects, getExperiments } from "@/content/projects";
import { cn } from "@/lib/utils";

export function ProjectGrid() {
  const featured = getFeaturedProjects();
  const experiments = getExperiments();

  const allTags = Array.from(
    new Set(featured.flatMap((p) => p.tags))
  ).sort();
  const [activeTag, setActiveTag] = useState<string | null>(null);

  const filtered = activeTag
    ? featured.filter((p) => p.tags.includes(activeTag))
    : featured;

  return (
    <div>
      {/* Tag filter */}
      <div className="flex flex-wrap justify-center gap-2 mb-10">
        <button
          onClick={() => setActiveTag(null)}
          className={cn(
            "px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
            !activeTag
              ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
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
              "px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
              tag === activeTag
                ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                : "bg-muted text-muted-foreground hover:text-foreground"
            )}
          >
            {tag}
          </button>
        ))}
      </div>

      {/* Featured grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {filtered.map((project, i) => (
          <ProjectCard key={project.slug} project={project} index={i} />
        ))}
      </div>

      {filtered.length === 0 && (
        <p className="text-center text-muted-foreground py-12">
          No projects match that filter.
        </p>
      )}

      {/* Experiments section */}
      {experiments.length > 0 && (
        <>
          <Separator className="bg-border my-16" />

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-10"
          >
            <div className="flex items-center justify-center gap-2 mb-3">
              <Beaker className="h-5 w-5 text-cyan-400" />
              <h2 className="text-2xl font-bold tracking-tight">Experiments</h2>
            </div>
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              Side projects and interactive toys — small-scope explorations
              built for fun and learning.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
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
