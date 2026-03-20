import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft, AlertTriangle, CheckCircle2, Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { TriageFlowDiagram } from "@/components/case-studies/triage-flow-diagram";

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
          <h2 className="mb-3 text-xl font-semibold">Key Tradeoff</h2>
          <p className="text-sm leading-relaxed text-muted-foreground">
            The consistent tradeoff was speed versus reliability. Fast,
            one-off fixes could close tickets quickly but often increased
            repeat incidents. A structured, reproducible triage path required
            more discipline up front, but produced better long-term resolution
            quality and lower ambiguity on future cases.
          </p>
        </section>

        <section className="rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            <h2 className="text-xl font-semibold">Outcome Signal</h2>
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
