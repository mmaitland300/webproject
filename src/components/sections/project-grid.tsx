"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { ProjectCard } from "@/components/sections/project-card";
import { projects } from "@/content/projects";
import { cn } from "@/lib/utils";

export function ProjectGrid() {
  const allTags = Array.from(new Set(projects.flatMap((p) => p.tags))).sort();
  const [activeTag, setActiveTag] = useState<string | null>(null);

  const filtered = activeTag
    ? projects.filter((p) => p.tags.includes(activeTag))
    : projects;

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

      {/* Grid */}
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
    </div>
  );
}
