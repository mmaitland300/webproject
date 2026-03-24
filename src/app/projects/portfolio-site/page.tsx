import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { ArrowLeft, CheckCircle2, FileCode2, Shield } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const PORTFOLIO_ARTIFACT_SRC = "/images/projects/portfolio-delivery-artifact.svg";

export const metadata: Metadata = {
  title: "Portfolio Website: Engineering and Operational Choices",
  description:
    "Engineering notes for this Next.js portfolio: typed content, contact form with validation and rate limiting, optional admin inbox behind GitHub OAuth, and pragmatic tradeoffs for a solo-maintained site.",
};

const safeguards = [
  "Server-side Zod validation for contact and waitlist before side effects",
  "Honeypot field plus Upstash sliding-window rate limiting on contact submissions",
  "GitHub OAuth via Auth.js for admin routes and inbox views when configured",
  "Optional Prisma persistence for contact and waitlist records after email sends",
  "Playwright smoke tests on public routes, titles, and OG metadata",
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
    title: "Small, reviewable changes",
    detail:
      "Incremental PRs and merge gates so fixes stay easy to reason about instead of landing as one large batch.",
  },
];

const evidenceLinks = [
  {
    label: "Contact form decision record",
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
            <Badge variant="secondary">Contact &amp; admin</Badge>
          </div>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Portfolio Website: Engineering and Operational Choices
          </h1>
          <p className="mt-4 max-w-3xl text-muted-foreground">
            This page documents how mmaitland.dev handles contact submissions,
            optional admin access, and content delivery — the same problems as
            any small production site, with choices tuned for one maintainer and
            low operational overhead.
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
                alt="Flow diagram: public routes, server actions, contact validation and rate limits, optional admin inbox"
                fill
                unoptimized
                className="object-contain object-center p-2 sm:p-4"
                sizes="(max-width: 768px) 100vw, 896px"
                priority
              />
            </div>
            <figcaption className="border-t border-border bg-card/50 px-4 py-3 text-center text-xs leading-relaxed text-muted-foreground">
              Sketch of the contact path and optional admin persistence — useful
              for keeping assumptions explicit when iterating alone.
            </figcaption>
          </figure>
        </section>

        <section className="mb-10 rounded-xl border border-border bg-card/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <Shield className="h-5 w-5 text-purple-400" />
            <h2 className="text-xl font-semibold">What runs in production</h2>
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
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            <h2 className="text-xl font-semibold">Where it stands</h2>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            Live at mmaitland.dev: contact works with validation and rate limiting,
            admin is gated when env is set, and decisions are documented in linked
            posts. Still iterating like any personal site.
          </p>
        </section>
      </div>
    </div>
  );
}
