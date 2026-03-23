import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { ArrowLeft, CheckCircle2, FileCode2, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const PORTFOLIO_ARTIFACT_SRC = "/images/projects/portfolio-delivery-artifact.svg";

export const metadata: Metadata = {
  title: "Portfolio Website Case Study",
  description:
    "How this Next.js portfolio is delivered with typed content, protected admin routes, and abuse-resistant contact handling under practical constraints.",
};

const safeguards = [
  "Server-side zod validation for contact and waitlist payloads before side effects",
  "Honeypot plus Upstash sliding-window limiting on inbound actions",
  "Auth.js GitHub OAuth gate for admin routes and inbox views",
  "Prisma-backed persistence for contact/waitlist records",
  "Playwright smoke coverage on all public routes with metadata and OG verification",
];

const tradeoffs = [
  {
    title: "Managed services over custom infrastructure",
    detail:
      "Used Resend, Upstash, and Neon-backed Prisma integration to reduce ops burden and improve reliability during solo iteration.",
  },
  {
    title: "Delivery first, then persistence",
    detail:
      "Contact action treats email delivery as the primary success path, with inbox persistence as best-effort to avoid silent form success when mail is misconfigured.",
  },
  {
    title: "Small-PR release discipline",
    detail:
      "Adopted merge-gated incremental updates to keep trust paths reviewable and prevent large, risky polish batches.",
  },
];

const evidenceLinks = [
  {
    label: "Contact pipeline decision record",
    href: "/blog/contact-pipeline-decision-record",
  },
  {
    label: "Repository (webproject)",
    href: "https://github.com/mmaitland300/webproject",
  },
  {
    label: "Playwright route smoke spec",
    href: "https://github.com/mmaitland300/webproject/blob/master/e2e/routes.spec.ts",
  },
];

export default function PortfolioSiteCaseStudyPage() {
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
            <Badge variant="secondary">Next.js</Badge>
            <Badge variant="secondary">Auth.js</Badge>
            <Badge variant="secondary">Prisma</Badge>
            <Badge variant="secondary">Operational Safeguards</Badge>
          </div>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Portfolio Website: delivery and trust-path case study
          </h1>
          <p className="mt-4 max-w-3xl text-muted-foreground">
            This page documents the engineering choices behind mmaitland.dev:
            how contact intake, admin access, and content delivery are structured
            to stay reliable under practical constraints rather than optimistic
            assumptions.
          </p>
        </header>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <FileCode2 className="h-5 w-5 text-cyan-400" />
            <h2 className="text-xl font-semibold">Architecture artifact</h2>
          </div>
          <figure className="overflow-hidden rounded-lg border border-border bg-muted/20">
            <div className="relative aspect-[1200/675] w-full">
              <Image
                src={PORTFOLIO_ARTIFACT_SRC}
                alt="Portfolio delivery flow showing public routes, server actions, abuse controls, and admin inbox path"
                fill
                unoptimized
                className="object-contain object-center p-2 sm:p-4"
                sizes="(max-width: 768px) 100vw, 896px"
                priority
              />
            </div>
            <figcaption className="border-t border-border bg-card/50 px-4 py-3 text-center text-xs leading-relaxed text-muted-foreground">
              Delivery artifact used to keep reliability assumptions explicit:
              input validation, abuse controls, authenticated admin access, and
              route-level regression checks.
            </figcaption>
          </figure>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-purple-400" />
            <h2 className="text-xl font-semibold">Operational safeguards</h2>
          </div>
          <ul className="space-y-2">
            {safeguards.map((item) => (
              <li
                key={item}
                className="rounded-lg border border-border bg-card/30 px-4 py-3 text-sm text-muted-foreground"
              >
                {item}
              </li>
            ))}
          </ul>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <h2 className="mb-3 text-xl font-semibold">Tradeoffs</h2>
          <div className="space-y-4">
            {tradeoffs.map((item) => (
              <div key={item.title}>
                <h3 className="text-sm font-medium text-foreground">{item.title}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{item.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <h2 className="mb-3 text-xl font-semibold">Evidence links</h2>
          <div className="space-y-2">
            {evidenceLinks.map((item) => {
              const isExternal = item.href.startsWith("http");
              return (
                <a
                  key={item.label}
                  href={item.href}
                  target={isExternal ? "_blank" : undefined}
                  rel={isExternal ? "noopener noreferrer" : undefined}
                  className="block rounded-lg border border-border bg-card/30 px-4 py-3 text-sm text-muted-foreground transition-colors hover:text-foreground"
                >
                  {item.label}
                </a>
              );
            })}
          </div>
        </section>

        <section className="rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            <h2 className="text-xl font-semibold">Where it stands</h2>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            The site reliably receives contact, protects admin surfaces behind
            OAuth, and documents its own engineering decisions through linked blog
            posts and case studies. It&apos;s live, tested, and under active iteration.
          </p>
        </section>
      </div>
    </div>
  );
}
