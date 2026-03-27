import type { Metadata } from "next";
import { AboutContent } from "@/components/sections/about-content";
import { SectionHeader } from "@/components/ui/section-header";
import { getPublicContactEmail } from "@/lib/site-contact";

export const metadata: Metadata = {
  title: "About",
  description:
    "How I diagnose multi-layer failures and apply that same method across web software, audio tooling, and production support.",
};

export default function AboutPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <SectionHeader
          eyebrow="About"
          title="Systems Diagnosis, Then Delivery"
          description="A method-first look at my work: observe, isolate, validate, and turn incident handling into repeatable engineering practice."
          className="mb-16"
        />
        <AboutContent publicEmail={getPublicContactEmail()} />
      </div>
    </div>
  );
}
