import Link from "next/link";

const proofItems = [
  {
    title: "Production simulator support under real constraints",
    detail:
      "Full-time incident triage at Auxillium for Full Swing GOLF software and simulator systems (2024-present).",
    href: "/projects/full-swing-tech-support",
  },
  {
    title: "Web delivery with auth and abuse controls",
    detail:
      "Live Next.js stack with admin OAuth, server-side validation, and contact intake protections.",
    href: "/blog/contact-pipeline-decision-record",
  },
  {
    title: "StringFlux DSP development",
    detail:
      "JUCE/C++ plugin work: multiband routing, transient-driven grain scheduling, and safe oversampling transitions.",
    href: "/projects/stringflux",
  },
  {
    title: "Core stack in active use",
    detail: "Next.js, TypeScript, Python, C++, PostgreSQL.",
  },
];

export function ProofStrip() {
  return (
    <section aria-label="Proof points" className="pb-8">
      <div className="mx-auto max-w-6xl px-6">
        <div className="grid gap-3 md:grid-cols-2">
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
