const proofItems = [
  {
    title: "Technical support in production environments",
    detail:
      "Full-time support at Auxillium for Full Swing GOLF software and simulator systems (2024-present).",
  },
  {
    title: "Live site with operational safeguards",
    detail:
      "Production Next.js site with admin auth, validated contact intake, and abuse controls.",
  },
  {
    title: "Audio software under active development",
    detail:
      "StringFlux JUCE/C++ plugin with multiband routing and transient-driven grain scheduling.",
  },
  {
    title: "Core stack",
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
              <h2 className="text-sm font-medium text-foreground">{item.title}</h2>
              <p className="mt-1 text-sm text-muted-foreground">{item.detail}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
