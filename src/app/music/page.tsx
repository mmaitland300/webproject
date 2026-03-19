import type { Metadata } from "next";
import { ExternalLink, Music2 } from "lucide-react";
import { SectionHeader } from "@/components/ui/section-header";

export const metadata: Metadata = {
  title: "Music",
  description: "Original music by Matt Maitland - listen on SoundCloud.",
};

const SOUNDCLOUD_PROFILE = "https://soundcloud.com/matthew_maitland";

const EMBED_PARAMS =
  "&color=%239333ea&auto_play=false&hide_related=true&show_comments=false&show_user=true&show_reposts=false&show_teaser=true&visual=true";

const tracks = [
  {
    title: "West-20d.mp3",
    artist: "NEUROCHEMICAL ENTROPY",
    trackUrl: "https://soundcloud.com/matthew_maitland/west-20d-mp3",
  },
  {
    title: "Gyroscopic Intuition - 7b",
    artist: "NEUROCHEMICAL ENTROPY",
    trackUrl: "https://soundcloud.com/matthew_maitland/gyroscopic-intuition-7b",
  },
  {
    title: "Stormy Saturday-2a",
    artist: "NEUROCHEMICAL ENTROPY",
    trackUrl: "https://soundcloud.com/matthew_maitland/renaissance",
  },
  {
    title: "(ex)Inhibitor - 4a",
    artist: "NEUROCHEMICAL ENTROPY",
    trackUrl: "https://soundcloud.com/matthew_maitland/exinhibitor-4a",
  },
];

export default function MusicPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <div className="mb-12">
          <SectionHeader
            eyebrow="Music"
            title="Audio Work and Experiments"
            description="I make music in my spare time. Here are a few tracks to start with, and you can head over to SoundCloud for the full catalog."
          />
          <div className="mt-5 flex justify-center">
            <Music2 className="h-6 w-6 text-[rgba(122,162,247,0.95)]" />
          </div>
        </div>

        <div className="space-y-10">
          {tracks.map((track) => (
            <section key={track.title} className="space-y-3">
              <div>
                <h2 className="text-lg font-semibold text-foreground">
                  {track.title}
                </h2>
                <p className="text-sm text-muted-foreground">{track.artist}</p>
              </div>

              <div className="overflow-hidden rounded-xl border border-border bg-card/50 backdrop-blur-sm">
                <iframe
                  width="100%"
                  height="300"
                  scrolling="no"
                  frameBorder="no"
                  allow="autoplay"
                  src={`https://w.soundcloud.com/player/?url=${encodeURIComponent(track.trackUrl)}${EMBED_PARAMS}`}
                  title={track.title}
                  className="block"
                />
              </div>

              <div className="text-sm text-muted-foreground">
                <a
                  href={track.trackUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="transition-colors hover:text-foreground"
                >
                  Open {track.title} on SoundCloud
                </a>
              </div>
            </section>
          ))}
        </div>

        <div className="mt-12 text-center">
          <a
            href={SOUNDCLOUD_PROFILE}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-md border border-border px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted"
          >
            <ExternalLink className="h-4 w-4" /> Follow on SoundCloud
          </a>
        </div>
      </div>
    </div>
  );
}
