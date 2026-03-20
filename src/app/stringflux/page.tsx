import type { Metadata } from "next";
import Link from "next/link";
import { Layers3, Zap, Music2, ArrowRight, Github } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { StringFluxPluginPreview } from "@/components/case-studies/stringflux-plugin-preview";
import { StringFluxWaitlistForm } from "@/components/sections/stringflux-waitlist-form";

export const metadata: Metadata = {
  title: "StringFlux",
  description:
    "StringFlux is a transient-aware multiband granular delay and freeze plugin for guitar and other stringed instruments. Join the waitlist for beta access.",
};

const howItWorks = [
  {
    icon: Music2,
    color: "text-purple-400",
    title: "You play",
    body: "StringFlux reads your input in real time. It tracks string attacks and captures a rolling history of your signal across three frequency bands.",
  },
  {
    icon: Zap,
    color: "text-cyan-400",
    title: "Grains fire on transients",
    body: "When you strike a note, a transient-driven scheduler fires grains timed to your playing. A density scheduler fills the space between attacks to build texture.",
  },
  {
    icon: Layers3,
    color: "text-amber-400",
    title: "Layered texture comes out",
    body: "The output blends your dry signal with the processed grain cloud — controllable density, grain length, pitch spread, and feedback. You stay in control.",
  },
];

export default function StringFluxPage() {
  return (
    <div className="py-24">
      <div className="mx-auto max-w-4xl px-6">

        {/* Hero */}
        <header className="mb-16 text-center">
          <div className="mb-4 flex flex-wrap justify-center gap-2">
            <Badge variant="secondary">Audio Plugin</Badge>
            <Badge variant="secondary">DSP</Badge>
            <Badge variant="secondary" className="border-amber-500/30 text-amber-400 bg-amber-500/10">
              In Development
            </Badge>
          </div>
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            StringFlux
          </h1>
          <p className="mt-4 mx-auto max-w-2xl text-lg text-muted-foreground leading-relaxed">
            Transient-aware multiband granular delay and freeze for guitar and
            stringed instruments. Turn a single performance into layered texture
            while keeping the response playable.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <a
              href="#waitlist"
              className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 px-5 py-2.5 text-sm font-medium text-white transition-all"
            >
              Join the waitlist <ArrowRight className="h-4 w-4" />
            </a>
            <a
              href="https://github.com/mmaitland300/StringFlux"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-border px-5 py-2.5 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="h-4 w-4" /> View source
            </a>
          </div>
        </header>

        {/* Proof artifact 1: Plugin UI */}
        <section className="mb-16" aria-labelledby="ui-section">
          <h2 id="ui-section" className="text-xl font-semibold mb-4">
            Plugin interface
          </h2>
          <StringFluxPluginPreview />
          <p className="mt-3 text-sm text-muted-foreground text-center">
            Current UI — work in progress. Knobs, band controls, and scheduler
            mode shown above are implemented and functional in the dev build.
          </p>
        </section>

        {/* Proof artifact 2: Audio demo */}
        <section className="mb-16 rounded-xl border border-border bg-card/40 p-6" aria-labelledby="audio-section">
          <h2 id="audio-section" className="text-xl font-semibold mb-2 flex items-center gap-2">
            <Music2 className="h-5 w-5 text-purple-400" />
            Audio demo
          </h2>
          <p className="text-sm text-muted-foreground mb-4">
            A short clip showing StringFlux processing a clean guitar signal — dry to
            fully wet with transient-triggered grain clouds.
          </p>
          {/* Placeholder: place stringflux-demo.mp3 at /public/audio/stringflux-demo.mp3 */}
          <audio
            controls
            className="w-full"
            aria-label="StringFlux audio demo"
          >
            <source src="/audio/stringflux-demo.mp3" type="audio/mpeg" />
            <source src="/audio/stringflux-demo.ogg" type="audio/ogg" />
            Your browser does not support the audio element.
          </audio>
          <p className="mt-3 text-xs text-muted-foreground">
            Demo clip coming soon — join the waitlist to be notified when it&apos;s available.
          </p>
        </section>

        {/* Proof artifact 3: How it works */}
        <section className="mb-16" aria-labelledby="how-section">
          <h2 id="how-section" className="text-xl font-semibold mb-6">
            How it works
          </h2>
          <div className="grid gap-4 sm:grid-cols-3">
            {howItWorks.map((step, i) => (
              <div
                key={step.title}
                className="rounded-xl border border-border bg-card/40 p-5"
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-semibold text-muted-foreground">
                    {i + 1}
                  </span>
                  <step.icon className={`h-4 w-4 ${step.color}`} />
                </div>
                <h3 className="font-semibold mb-1">{step.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {step.body}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Waitlist form */}
        <section
          id="waitlist"
          className="rounded-xl border border-purple-500/20 bg-card/60 p-8"
          aria-labelledby="waitlist-section"
        >
          <div className="max-w-lg">
            <h2
              id="waitlist-section"
              className="text-2xl font-bold mb-1"
            >
              Get notified when it&apos;s ready
            </h2>
            <p className="text-muted-foreground mb-6">
              StringFlux is in active development. Leave your email and I&apos;ll
              reach out when beta access or a first release is available.
            </p>
            <StringFluxWaitlistForm />
          </div>
        </section>

        {/* Footer link to technical case study */}
        <div className="mt-10 text-center">
          <Link
            href="/projects/stringflux"
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Read the DSP architecture case study <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>

      </div>
    </div>
  );
}
