"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ExternalLink, Github, Gamepad2, Play, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { Project } from "@/content/projects";

interface ProjectCardProps {
  project: Project;
  index: number;
  compact?: boolean;
}

const outcomeLabels: Record<
  NonNullable<Project["outcomeType"]>,
  string
> = {
  metric: "Outcome (Metric)",
  proxy: "Outcome (Operational Proxy)",
  technical: "Outcome (Technical)",
  qualitative: "Outcome (Qualitative)",
};

export function ProjectCard({ project, index, compact }: ProjectCardProps) {
  const [iframeActive, setIframeActive] = useState(false);

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
        <div className="relative h-48 bg-black/50 overflow-hidden">
          {iframeActive ? (
            <>
              <iframe
                src={project.iframe}
                className="w-full h-full border-0"
                title={project.title}
                sandbox="allow-scripts allow-same-origin"
              />
              <button
                onClick={() => setIframeActive(false)}
                className="absolute top-2 right-2 z-20 p-1 rounded-md bg-black/60 text-white/70 hover:text-white transition-colors"
                aria-label="Close interactive preview"
              >
                <X size={14} />
              </button>
            </>
          ) : (
            <button
              onClick={() => setIframeActive(true)}
              className="w-full h-full flex flex-col items-center justify-center gap-2 cursor-pointer"
            >
              <div className="p-3 rounded-full bg-purple-500/20 text-purple-400 transition-transform group-hover:scale-110">
                <Play size={24} className="ml-0.5" />
              </div>
              <span className="text-xs text-muted-foreground">
                Click to play
              </span>
            </button>
          )}
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

        {!compact && project.problem && (
          <div className="space-y-2 mb-4 text-sm">
            <div>
              <span className="font-medium text-foreground/80">Problem: </span>
              <span className="text-muted-foreground">{project.problem}</span>
            </div>
            {project.constraints && (
              <div>
                <span className="font-medium text-foreground/80">
                  Constraints:{" "}
                </span>
                <span className="text-muted-foreground">
                  {project.constraints}
                </span>
              </div>
            )}
            {project.tradeoff && (
              <div>
                <span className="font-medium text-foreground/80">
                  Tradeoff:{" "}
                </span>
                <span className="text-muted-foreground">{project.tradeoff}</span>
              </div>
            )}
            {project.role && (
              <div>
                <span className="font-medium text-foreground/80">Role: </span>
                <span className="text-muted-foreground">{project.role}</span>
              </div>
            )}
            {project.outcome && (
              <div>
                <span className="font-medium text-foreground/80">
                  {project.outcomeType
                    ? `${outcomeLabels[project.outcomeType]}: `
                    : "Outcome: "}
                </span>
                <span className="text-muted-foreground">{project.outcome}</span>
              </div>
            )}
          </div>
        )}

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
          {project.caseStudy && (
            <a
              href={project.caseStudy}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <ExternalLink size={14} /> Case Study
            </a>
          )}
          {project.iframe && (
            <button
              onClick={() => setIframeActive(!iframeActive)}
              className="flex items-center gap-1.5 text-xs text-purple-400 hover:text-purple-300 transition-colors"
            >
              <Gamepad2 size={14} /> {iframeActive ? "Stop" : "Play"}
            </button>
          )}
        </div>
      </div>
    </motion.article>
  );
}
