import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft, BarChart3, FlaskConical, ListChecks } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { MlEvalWorkflowDiagram } from "@/components/case-studies/ml-eval-workflow-diagram";

export const metadata: Metadata = {
  title: "Snake Detector (CNN) Case Study",
  description:
    "How dataset hygiene, fixed splits, and confusion-matrix review stay ahead of naive accuracy on a small snake photo dataset.",
};

const artifactTable = [
  {
    artifact: "Stratified train/val split",
    purpose: "Keep class ratios stable so metrics reflect generalization, not a lucky split.",
  },
  {
    artifact: "Augmentation policy (logged)",
    purpose: "Make image-level changes comparable across runs instead of silently drifting.",
  },
  {
    artifact: "Confusion matrix + per-class review",
    purpose: "Surface which species are confused before touching model depth or width.",
  },
  {
    artifact: "Run folder (config + metrics snapshot)",
    purpose: "Reproduce any reported number without guessing which code version produced it.",
  },
];

export default function SnakeDetectorCaseStudyPage() {
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
            <Badge variant="secondary">Computer Vision</Badge>
            <Badge variant="secondary">CNN</Badge>
            <Badge variant="secondary">Evaluation</Badge>
            <Badge variant="secondary">Experiment</Badge>
          </div>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Snake Detector: evaluation-first CNN experiment
          </h1>
          <p className="mt-4 max-w-3xl text-muted-foreground">
            This page is the engineering story behind the repo: a small, noisy
            classification problem where the main risk is fooling yourself with
            headline accuracy. The proof is in the workflow and artifacts, not a
            single leaderboard score.
          </p>
        </header>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <FlaskConical className="h-5 w-5 text-cyan-400" />
            <h2 className="text-xl font-semibold">Problem framing</h2>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            With limited images, class imbalance, and imperfect labels, the model
            can look fine on aggregate accuracy while failing on the species that
            matter for real use. The goal of this project was to keep every
            training run comparable and to force error analysis before chasing
            bigger architectures.
          </p>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <ListChecks className="h-5 w-5 text-purple-400" />
            <h2 className="text-xl font-semibold">Artifacts (what gets saved)</h2>
          </div>
          <MlEvalWorkflowDiagram />
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[480px] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-border text-xs uppercase tracking-wide text-muted-foreground">
                  <th className="py-2 pr-4 font-medium">Artifact</th>
                  <th className="py-2 font-medium">Purpose</th>
                </tr>
              </thead>
              <tbody>
                {artifactTable.map((row) => (
                  <tr key={row.artifact} className="border-b border-border/60">
                    <td className="py-3 pr-4 font-medium text-foreground">
                      {row.artifact}
                    </td>
                    <td className="py-3 text-muted-foreground">{row.purpose}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-amber-400" />
            <h2 className="text-xl font-semibold">Outcome signal</h2>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            <span className="font-medium text-foreground">Technical outcome:</span>{" "}
            a repeatable loop where poor classes show up in structured review
            instead of hiding behind a single accuracy number. The repo is the
            source of truth for scripts and training flow; plug in your own
            metrics exports or confusion matrices as you iterate.
          </p>
        </section>
      </div>
    </div>
  );
}
