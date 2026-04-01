import type { Metadata } from "next";
import { MainContentAnchor } from "@/components/layout/main-content-anchor";
import { ResumeDocument } from "@/components/resume/resume-document";
import { getPublicContactEmail } from "@/lib/site-contact";

export const metadata: Metadata = {
  title: "Resume",
  description:
    "Resume for Matt Maitland covering full-stack development and technical support experience.",
};

export default function ResumePage() {
  const publicEmail = getPublicContactEmail();
  return (
    <div className="py-24">
      <MainContentAnchor />
      <ResumeDocument variant="web" publicEmail={publicEmail} />
    </div>
  );
}
