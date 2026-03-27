import Link from "next/link";
import type { ReactNode } from "react";
import type { Project } from "@/content/projects";

const statusLabel: Record<NonNullable<Project["status"]>, string> = {
  "in-progress": "In Progress",
  operational: "Operational",
  shipped: "Shipped",
  archived: "Archived",
};

interface CaseStudyEvidenceFooterProps {
  project: Project;
  statusIcon: ReactNode;
}

export function CaseStudyEvidenceFooter({
  project,
  statusIcon,
}: CaseStudyEvidenceFooterProps) {
  return (
    <>
      <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
        <h2 className="mb-3 text-xl font-semibold">Evidence links</h2>
        <div className="space-y-2">
          {(project.proofLinks ?? []).map((item) => {
            const isExternal = item.href.startsWith("http");
            const className =
              "block rounded-lg border border-border bg-card/30 px-4 py-3 text-sm text-muted-foreground transition-colors hover:text-foreground";
            return isExternal ? (
              <a
                key={item.label}
                href={item.href}
                target="_blank"
                rel="noopener noreferrer"
                className={className}
              >
                {item.label}
              </a>
            ) : (
              <Link key={item.label} href={item.href} className={className}>
                {item.label}
              </Link>
            );
          })}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card/40 p-6">
        <div className="mb-3 flex items-center gap-2">
          {statusIcon}
          <h2 className="text-xl font-semibold">Where it stands</h2>
        </div>
        {project.status ? (
          <p className="mb-2 text-sm text-foreground/90">
            <span className="font-medium">Status:</span> {statusLabel[project.status]}
          </p>
        ) : null}
        {project.evidence ? (
          <p className="text-sm leading-relaxed text-muted-foreground">
            {project.evidence}
          </p>
        ) : null}
        {project.knownLimits ? (
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            <span className="font-medium text-foreground/90">Known limits:</span>{" "}
            {project.knownLimits}
          </p>
        ) : null}
      </section>
    </>
  );
}
