import Link from "next/link";

const proofItems = [
  {
    title: "Full Swing simulator support",
    detail:
      "Remote triage across calibration, licensing, display, and networking failures at Auxillium (2024-present). Documented as a case study with failure patterns and triage methodology.",
    href: "/projects/full-swing-tech-support",
  },
  {
    title: "This site: contact form and admin auth",
    detail:
      "Next.js 16, Zod validation, rate limiting, GitHub OAuth admin. Engineering choices documented in a public decision record.",
    href: "/blog/contact-pipeline-decision-record",
  },
  {
    title: "StringFlux: JUCE/C++ audio plugin",
    detail:
      "Multiband granular delay for guitar with transient-aware scheduling and safe oversampling transitions. Architecture documented in a DSP case study.",
    href: "/projects/stringflux",
  },
];

export function ProofStrip() {
  return (
    <section aria-label="Proof points" className="pb-8">
      <div className="mx-auto max-w-6xl px-6">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {proofItems.map((item) => (
            <article
              key={item.title}
              className="rounded-lg border border-border bg-card/40 px-4 py-3"
            >
              {"href" in item && item.href ? (
                <Link
                  href={item.href}
                  className="text-sm font-medium text-foreground hover:text-purple-400 transition-colors"
                >
                  {item.title}
                </Link>
              ) : (
                <h2 className="text-sm font-medium text-foreground">
                  {item.title}
                </h2>
              )}
              <p className="mt-1 text-sm text-muted-foreground">{item.detail}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
