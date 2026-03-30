import type { Metadata } from "next";
import { ResumeDocument } from "@/components/resume/resume-document";
import { getPublicContactEmail } from "@/lib/site-contact";

export const metadata: Metadata = {
  title: "Resume (print)",
  description:
    "Print-friendly resume layout for PDF export. Same content as the public resume page.",
  robots: {
    index: false,
    follow: false,
  },
};

export default function ResumePrintPage() {
  const publicEmail = getPublicContactEmail();
  return <ResumeDocument variant="print" publicEmail={publicEmail} />;
}
