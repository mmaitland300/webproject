"use client";

import { motion } from "framer-motion";
import { ExternalLink, Github, Gamepad2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { Project } from "@/content/projects";

interface ProjectCardProps {
  project: Project;
  index: number;
}

export function ProjectCard({ project, index }: ProjectCardProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className="group relative rounded-xl border border-border bg-card/50 backdrop-blur-sm overflow-hidden hover:border-purple-500/30 transition-all duration-300"
    >
      {/* Preview area */}
      {project.iframe ? (
        <div className="relative h-48 bg-black/50 flex items-center justify-center overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent to-card/80 z-10 pointer-events-none group-hover:opacity-0 transition-opacity duration-300" />
          <Gamepad2 className="h-10 w-10 text-muted-foreground/30 group-hover:opacity-0 transition-opacity" />
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
            <iframe
              src={project.iframe}
              className="w-full h-full border-0"
              title={project.title}
              loading="lazy"
              sandbox="allow-scripts allow-same-origin"
            />
          </div>
        </div>
      ) : (
        <div className="h-48 bg-gradient-to-br from-purple-500/10 to-cyan-500/10 flex items-center justify-center">
          <div className="text-4xl font-bold gradient-text opacity-30">
            {project.title[0]}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="p-6">
        <h3 className="text-lg font-semibold text-foreground mb-2 group-hover:text-purple-400 transition-colors">
          {project.title}
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed mb-4">
          {project.description}
        </p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mb-4">
          {project.tags.map((tag) => (
            <Badge
              key={tag}
              variant="secondary"
              className="text-xs font-normal"
            >
              {tag}
            </Badge>
          ))}
        </div>

        {/* Links */}
        <div className="flex items-center gap-3">
          {project.github && (
            <a
              href={project.github}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github size={14} /> Code
            </a>
          )}
          {project.demo && (
            <a
              href={project.demo}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <ExternalLink size={14} /> Demo
            </a>
          )}
          {project.iframe && (
            <span className="flex items-center gap-1.5 text-xs text-purple-400">
              <Gamepad2 size={14} /> Hover to play
            </span>
          )}
        </div>
      </div>
    </motion.article>
  );
}
