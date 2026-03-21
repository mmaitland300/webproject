"use client";

import { useState } from "react";
import { Play } from "lucide-react";

interface LazySoundCloudTrackProps {
  title: string;
  trackUrl: string;
  embedParams: string;
}

export function LazySoundCloudTrack({
  title,
  trackUrl,
  embedParams,
}: LazySoundCloudTrackProps) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card/50 backdrop-blur-sm">
      {loaded ? (
        <iframe
          width="100%"
          height="300"
          scrolling="no"
          frameBorder="no"
          allow="autoplay"
          src={`https://w.soundcloud.com/player/?url=${encodeURIComponent(trackUrl)}${embedParams}`}
          title={title}
          className="block"
        />
      ) : (
        <button
          type="button"
          onClick={() => setLoaded(true)}
          className="flex h-[300px] w-full flex-col items-center justify-center gap-3 text-center"
          aria-label={`Load SoundCloud player for ${title}`}
        >
          <span className="rounded-full bg-purple-500/20 p-3 text-purple-300">
            <Play className="h-5 w-5" />
          </span>
          <span className="text-sm font-medium text-foreground">
            Load player for {title}
          </span>
          <span className="max-w-xs text-xs text-muted-foreground">
            Click to load the SoundCloud embed only when needed.
          </span>
        </button>
      )}
    </div>
  );
}
