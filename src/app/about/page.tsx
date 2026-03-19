import type { Metadata } from "next";
import { AboutContent } from "@/components/sections/about-content";

export const metadata: Metadata = {
  title: "About",
  description:
    "Learn about my background, skills, and experience as a full-stack developer.",
};

export default function AboutPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <div className="text-center mb-16">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
            About <span className="gradient-text">Me</span>
          </h1>
          <p className="mt-4 text-muted-foreground max-w-lg mx-auto">
            Developer, builder, and lifelong learner.
          </p>
        </div>
        <AboutContent />
      </div>
    </div>
  );
}
