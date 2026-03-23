import Link from "next/link";

const proofItems = [
  {
    title: "Production simulator support under real constraints",
    detail:
      "Full-time incident triage at Auxillium for Full Swing simulator systems — hardware, software, networking, and OS layers (2024–present).",
    href: "/projects/full-swing-tech-support",
  },
  {
    title: "Web delivery with auth and abuse controls",
    detail:
      "This site: Next.js 16, server-side validation, honeypot + Redis rate limiting, GitHub OAuth admin inbox. Documented in a public decision record.",
    href: "/blog/contact-pipeline-decision-record",
  },
  {
    title: "StringFlux — audio plugin in active development",
    detail:
      "JUCE/C++ multiband granular delay with transient-driven grain scheduling and real-time safe oversampling transitions.",
    href: "/projects/stringflux",
  },
  {
    title: "ML evaluation discipline",
    detail:
      "CNN snake classifier focused on dataset hygiene, stratified splits, and confusion-matrix-driven review before architecture changes.",
    href: "/projects/snake-detector",
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
