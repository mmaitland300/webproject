import type { Metadata } from "next";
import { AboutContent } from "@/components/sections/about-content";
import { SectionHeader } from "@/components/ui/section-header";

export const metadata: Metadata = {
  title: "About",
  description:
    "Learn about my background, skills, and experience as a full-stack developer.",
};

export default function AboutPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <SectionHeader
          eyebrow="About"
          title="Developer, Builder, Problem Solver"
          description="A focused look at my technical background, hands-on support experience, and the tools I use to ship practical work."
          className="mb-16"
        />
        <AboutContent />
      </div>
    </div>
  );
}
