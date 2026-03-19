import type { Metadata } from "next";
import { ProjectGrid } from "@/components/sections/project-grid";
import { SectionHeader } from "@/components/ui/section-header";

export const metadata: Metadata = {
  title: "Projects",
  description:
    "Professional work, side projects, and interactive experiments by Matt Maitland.",
};

export default function ProjectsPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-6xl px-6">
        <SectionHeader
          eyebrow="Projects"
          title="Work Worth Showing"
          description="Professional work, side projects, and interactive experiments that reflect how I build, troubleshoot, and iterate."
          className="mb-12"
        />
        <ProjectGrid />
      </div>
    </div>
  );
}
