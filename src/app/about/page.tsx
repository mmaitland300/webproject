import type { Metadata } from "next";
import { AboutContent } from "@/components/sections/about-content";
import { SectionHeader } from "@/components/ui/section-header";
import { getPublicContactEmail } from "@/lib/site-contact";

export const metadata: Metadata = {
  title: "About",
  description:
    "Background across web applications, audio software, and remote technical support in real customer environments.",
};

export default function AboutPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <SectionHeader
          eyebrow="About"
          title="Developer, Builder, Technical Support"
          description="A closer look at my background across software development, audio tooling, and remote troubleshooting work in production environments."
          className="mb-16"
        />
        <AboutContent publicEmail={getPublicContactEmail()} />
      </div>
    </div>
  );
}
