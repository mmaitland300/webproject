import type { Metadata } from "next";
import { ProjectGrid } from "@/components/sections/project-grid";
import { SectionHeader } from "@/components/ui/section-header";

export const metadata: Metadata = {
  title: "Projects",
  description:
    "Featured case studies with constraints, evidence, and limits. Smaller experiments are listed separately so learning builds do not dilute the main signal.",
};

export default function ProjectsPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-6xl px-6">
        <SectionHeader
          eyebrow="Projects"
          title="Case Studies and Experiments"
          description="Featured case studies first: explicit constraints, decisions, and proof. Experiments and smaller builds are grouped below so the strongest signal stays easy to scan."
          className="mb-12"
        />
        <ProjectGrid />
      </div>
    </div>
  );
}
