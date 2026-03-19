import type { Metadata } from "next";
import { ProjectGrid } from "@/components/sections/project-grid";

export const metadata: Metadata = {
  title: "Projects",
  description:
    "Browse my projects — interactive simulations, games, and web applications.",
};

export default function ProjectsPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-6xl px-6">
        <div className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
            My <span className="gradient-text">Projects</span>
          </h1>
          <p className="mt-4 text-muted-foreground max-w-lg mx-auto">
            Things I&apos;ve built, from quick experiments to full applications.
            Hover over game cards to play them right in the browser.
          </p>
        </div>
        <ProjectGrid />
      </div>
    </div>
  );
}
