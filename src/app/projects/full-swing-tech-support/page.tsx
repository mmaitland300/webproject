import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { ArrowLeft, AlertTriangle, CheckCircle2, Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { TriageFlowDiagram } from "@/components/case-studies/triage-flow-diagram";

/** Same asset as the Full Swing project card thumbnail. */
const TRIAGE_ARTIFACT_SRC = "/images/projects/full-swing-triage-artifact.svg";

export const metadata: Metadata = {
  title: "Full Swing Technical Support Case Study",
  description:
    "A technical troubleshooting case study covering triage workflows, failure isolation, and decision tradeoffs in Full Swing simulator environments.",
};

const failureModes = [
  "Calibration drift impacting ball/club data accuracy",
  "Licensing and activation failures after environment changes",
  "Display and graphics mismatches across GPU/OS updates",
  "Network/configuration instability affecting software behavior",
  "Peripheral and Windows conflicts causing intermittent faults",
];

const workflow = [
  {
    step: "1. Scope and classify symptoms",
    detail:
      "Separate user-reported symptoms from underlying system layers (hardware, licensing, network, OS, graphics) to avoid premature root-cause assumptions.",
  },
  {
    step: "2. Build a reproducible baseline",
    detail:
      "Capture current environment state and reproduce under controlled conditions to reduce noise and identify which variables are actually causal.",
  },
  {
    step: "3. Isolate likely failure branches",
    detail:
      "Use branch-based diagnostics rather than ad hoc guessing: test one subsystem at a time and eliminate competing hypotheses with evidence.",
  },
  {
    step: "4. Apply lowest-risk corrective action",
    detail:
      "Prioritize reversible fixes first, then progress to deeper remediations only when diagnostics confirm they are necessary.",
  },
  {
    step: "5. Document and codify pattern",
    detail:
      "Convert resolved incidents into repeatable troubleshooting paths so similar issues are solved faster and with higher consistency.",
  },
];

const representativeIncident = [
  {
    label: "Symptom",
    detail:
      "Customer reports inconsistent launch and spin reads after a software/OS update window, with occasional sessions that appear normal and then degrade.",
  },
  {
    label: "Candidate causes",
    detail:
      "Calibration state drift, graphics/driver mismatch, peripheral input instability, and environment changes introduced during updates.",
  },
  {
    label: "Branch elimination path",
    detail:
      "Reproduce in a controlled baseline, then isolate one subsystem at a time (calibration checks, display pipeline checks, peripheral checks) to remove non-causal branches.",
  },
  {
    label: "Root cause pattern",
    detail:
      "A mixed-state configuration where calibration assumptions and post-update environment behavior no longer matched clean baseline expectations.",
  },
  {
    label: "Fix pattern",
    detail:
      "Apply reversible corrective actions in order (restore known-good settings, recalibrate in sequence, re-verify environment) before deeper changes.",
  },
  {
    label: "What changed after",
    detail:
      "Issue handling became faster on similar escalations because the incident was converted into a repeatable checklist rather than a one-off memory.",
  },
];

const evidenceLinks = [
  {
    label: "Troubleshooting playbook post (real incident format)",
    href: "/blog/troubleshooting-playbook-multi-layer-failures",
  },
  {
    label: "Projects overview entry",
    href: "/projects",
  },
  {
    label: "Resume role context",
    href: "/resume",
  },
];

export default function FullSwingCaseStudyPage() {
  return (
    <div className="py-24">
      <div className="mx-auto max-w-4xl px-6">
        <Link
          href="/projects"
          className="mb-8 inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft size={14} /> Back to projects
        </Link>

        <header className="mb-12">
          <div className="mb-4 flex flex-wrap gap-2">
            <Badge variant="secondary">Technical Support</Badge>
            <Badge variant="secondary">Troubleshooting</Badge>
            <Badge variant="secondary">Failure Analysis</Badge>
            <Badge variant="secondary">Operational Workflow</Badge>
          </div>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Full Swing Technical Support Case Study
          </h1>
          <p className="mt-4 max-w-3xl text-muted-foreground">
            This case study focuses on how multi-layer simulator issues were
            diagnosed and resolved under real constraints, where hardware,
            software, networking, and operating-system behavior frequently
            overlapped.
          </p>
        </header>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-400" />
            <h2 className="text-xl font-semibold">Problem Context</h2>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            Incidents were rarely single-point failures. Most escalations
            involved multiple plausible causes and partial signals, creating a
            high risk of quick but unstable fixes. The key challenge was
            identifying root cause with enough confidence to prevent recurrence.
          </p>
        </section>

        <section className="mb-10">
          <h2 className="mb-4 text-xl font-semibold">Common Failure Modes</h2>
          <ul className="space-y-2">
            {failureModes.map((mode) => (
              <li
                key={mode}
                className="rounded-lg border border-border bg-card/30 px-4 py-3 text-sm text-muted-foreground"
              >
                {mode}
              </li>
            ))}
          </ul>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <Wrench className="h-5 w-5 text-cyan-400" />
            <h2 className="text-xl font-semibold">Diagnostic Workflow</h2>
          </div>
          <figure className="mb-8 overflow-hidden rounded-lg border border-border bg-muted/20">
            <div className="relative aspect-[1200/675] w-full">
              <Image
                src={TRIAGE_ARTIFACT_SRC}
                alt="Branch-based triage workflow: classify symptoms, build baseline, branch testing, apply fix, verify, and document"
                fill
                unoptimized
                className="object-contain object-center p-2 sm:p-4"
                sizes="(max-width: 768px) 100vw, 896px"
                priority
              />
            </div>
            <figcaption className="border-t border-border bg-card/50 px-4 py-3 text-center text-xs leading-relaxed text-muted-foreground">
              Same overview as the project card preview: a representative
              branch-based triage artifact for multi-layer simulator support
              (not an official Full Swing diagram).
            </figcaption>
          </figure>
          <p className="mb-4 text-sm text-muted-foreground">
            Below is a simplified linear view of the same idea, useful when you
            want a quick read of the end-to-end path.
          </p>
          <TriageFlowDiagram />
          <div className="mt-8 space-y-4">
            {workflow.map((item) => (
              <div key={item.step}>
                <h3 className="text-sm font-medium text-foreground">{item.step}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{item.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <h2 className="mb-2 text-xl font-semibold">
            Representative Incident Pattern
          </h2>
          <p className="mb-4 text-sm leading-relaxed text-muted-foreground">
            This walkthrough reflects a recurring class of escalations, not a
            single customer transcript. Details are intentionally anonymized and
            scoped to troubleshooting patterns I can defend publicly.
          </p>
          <div className="space-y-3">
            {representativeIncident.map((item) => (
              <div
                key={item.label}
                className="rounded-lg border border-border bg-card/30 px-4 py-3 text-sm"
              >
                <span className="font-medium text-foreground">{item.label}: </span>
                <span className="text-muted-foreground">{item.detail}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <h2 className="mb-3 text-xl font-semibold">Tradeoffs</h2>
          <p className="text-sm leading-relaxed text-muted-foreground">
            The consistent tradeoff was speed versus reliability. Fast,
            one-off fixes could close tickets quickly but often increased
            repeat incidents. A structured, reproducible triage path required
            more discipline up front, but produced better long-term resolution
            quality and lower ambiguity on future cases.
          </p>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <h2 className="mb-3 text-xl font-semibold">Evidence links</h2>
          <div className="space-y-2">
            {evidenceLinks.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className="block rounded-lg border border-border bg-card/30 px-4 py-3 text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                {item.label}
              </Link>
            ))}
          </div>
        </section>

        <section className="rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            <h2 className="text-xl font-semibold">Current state</h2>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            <span className="font-medium text-foreground">
              Qualitative outcome:
            </span>{" "}
            issue resolution became more consistent by standardizing
            troubleshooting paths, isolating root causes across multiple
            plausible failures, and reducing reliance on non-reproducible
            one-off fixes.
          </p>
        </section>
      </div>
    </div>
  );
}
