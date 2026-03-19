"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { ProjectCard } from "@/components/sections/project-card";
import { getFeaturedProjects } from "@/content/projects";

export function FeaturedProjects() {
  const featured = getFeaturedProjects().slice(0, 3);

  return (
    <section className="py-24 relative z-10">
      <div className="mx-auto max-w-6xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
            Featured{" "}
            <span className="gradient-text">Projects</span>
          </h2>
          <p className="mt-3 text-muted-foreground max-w-lg mx-auto">
            Professional work and selected side projects &mdash; from
            client applications to interactive experiments.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {featured.map((project, i) => (
            <ProjectCard key={project.slug} project={project} index={i} />
          ))}
        </div>

        <div className="mt-10 text-center">
          <Link
            href="/projects"
            className={buttonVariants({ variant: "outline" })}
          >
            View All Projects <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>
      </div>
    </section>
  );
}
