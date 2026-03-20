/**
 * Stylized render of the StringFlux plugin UI for the product landing page.
 * This is a designed visual representation -- not a screenshot -- showing the
 * intended layout: header, waveform display, band controls, and knob row.
 */

interface KnobProps {
  cx: number;
  cy: number;
  r: number;
  value: number; // 0-1
  color: string;
  label: string;
}

function Knob({ cx, cy, r, value, color, label }: KnobProps) {
  const angle = -135 + value * 270;
  const rad = (angle * Math.PI) / 180;
  const ix = cx + (r - 3) * Math.sin(rad);
  const iy = cy - (r - 3) * Math.cos(rad);

  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill="#27272a" stroke="#3f3f46" strokeWidth="1.5" />
      <circle cx={cx} cy={cy} r={r - 3} fill="#1c1c1f" stroke="none" />
      <line
        x1={cx}
        y1={cy}
        x2={ix}
        y2={iy}
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx={cx} cy={cy} r={2} fill={color} />
      <text
        x={cx}
        y={cy + r + 12}
        textAnchor="middle"
        fill="#71717a"
        fontSize="7"
        fontFamily="monospace"
        letterSpacing="0.5"
      >
        {label.toUpperCase()}
      </text>
    </g>
  );
}

function Band({
  x,
  y,
  width,
  label,
  color,
  active,
}: {
  x: number;
  y: number;
  width: number;
  label: string;
  color: string;
  active?: boolean;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={28}
        rx="4"
        fill={active ? `${color}22` : "#18181b"}
        stroke={active ? color : "#3f3f46"}
        strokeWidth="1"
      />
      <text
        x={x + width / 2}
        y={y + 16}
        textAnchor="middle"
        fill={active ? color : "#71717a"}
        fontSize="8"
        fontFamily="monospace"
        letterSpacing="1"
      >
        {label}
      </text>
    </g>
  );
}

function WaveformPath({
  x,
  y,
  width,
  height,
  color,
  seed,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  seed: number;
}) {
  // Deterministic waveform shape seeded to always render the same
  const points: [number, number][] = [];
  const steps = 48;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const px = x + t * width;
    const wave =
      Math.sin((t * 12 + seed) * Math.PI) * 0.6 +
      Math.sin((t * 7 + seed * 1.3) * Math.PI) * 0.3 +
      Math.sin((t * 3 + seed * 0.7) * Math.PI) * 0.15;
    const py = y + height / 2 - (wave * height) / 2;
    points.push([px, py]);
  }
  const d = points
    .map(([px, py], i) => `${i === 0 ? "M" : "L"} ${px.toFixed(1)} ${py.toFixed(1)}`)
    .join(" ");
  return <path d={d} fill="none" stroke={color} strokeWidth="1.5" opacity="0.8" />;
}

export function StringFluxPluginPreview() {
  const W = 680;
  const H = 240;

  return (
    <figure className="overflow-x-auto rounded-xl border border-border bg-zinc-950 p-2">
      <figcaption className="mb-2 text-center text-xs text-muted-foreground font-mono">
        StringFlux — plugin UI (work in progress)
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto h-auto w-full max-w-3xl"
        role="img"
        aria-label="StringFlux plugin interface showing header, waveform display, band controls, and knob row for density, grain length, pitch, feedback, and mix"
      >
        {/* Plugin frame */}
        <rect x="0" y="0" width={W} height={H} rx="8" fill="#0f0f11" stroke="#27272a" strokeWidth="1.5" />

        {/* Header bar */}
        <rect x="0" y="0" width={W} height="36" rx="8" fill="#18181b" />
        <rect x="0" y="28" width={W} height="8" fill="#18181b" />
        <rect x="0" y="34" width={W} height="2" fill="#27272a" />

        {/* Plugin name */}
        <text x="16" y="22" fill="#e4e4e7" fontSize="13" fontWeight="bold" fontFamily="monospace" letterSpacing="3">
          STRINGFLUX
        </text>
        <text x="154" y="22" fill="#a855f7" fontSize="8" fontFamily="monospace" letterSpacing="1">
          GRANULAR DELAY + FREEZE
        </text>

        {/* Oversampling selector */}
        {[["1x", false], ["2x", true], ["4x", false]].map(([label, active], i) => (
          <g key={String(label)}>
            <rect
              x={W - 88 + i * 28}
              y="9"
              width="24"
              height="18"
              rx="3"
              fill={active ? "#1e1b4b" : "#27272a"}
              stroke={active ? "#818cf8" : "#3f3f46"}
              strokeWidth="1"
            />
            <text
              x={W - 88 + i * 28 + 12}
              y="21"
              textAnchor="middle"
              fill={active ? "#818cf8" : "#71717a"}
              fontSize="8"
              fontFamily="monospace"
            >
              {label}
            </text>
          </g>
        ))}

        {/* Waveform display area */}
        <rect x="12" y="44" width={W - 24} height="72" rx="6" fill="#0c0c0e" stroke="#27272a" strokeWidth="1" />

        {/* Waveform traces — transient + grain clouds */}
        <WaveformPath x={14} y={44} width={W - 28} height={72} color="#a855f7" seed={1.2} />
        <WaveformPath x={14} y={44} width={W - 28} height={72} color="#22d3ee" seed={2.7} />

        {/* Freeze indicator */}
        <rect x={W - 80} y="50" width="68" height="20" rx="4" fill="#0f172a" stroke="#1d4ed8" strokeWidth="1" />
        <circle cx={W - 69} cy="60" r="4" fill="#3b82f6" opacity="0.7" />
        <text x={W - 62} y="64" fill="#93c5fd" fontSize="8" fontFamily="monospace">FREEZE</text>

        {/* Band controls row */}
        <Band x={12} y={124} width={102} label="LO BAND" color="#22d3ee" active />
        <Band x={120} y={124} width={102} label="MID BAND" color="#a855f7" active />
        <Band x={228} y={124} width={102} label="HI BAND" color="#f59e0b" />

        {/* Scheduler mode */}
        <rect x={338} y={124} width={100} height={28} rx="4" fill="#1a1a1f" stroke="#3f3f46" strokeWidth="1" />
        <text x={388} y={136} textAnchor="middle" fill="#71717a" fontSize="7" fontFamily="monospace" letterSpacing="0.5">SCHEDULER</text>
        <text x={388} y={146} textAnchor="middle" fill="#a855f7" fontSize="8" fontFamily="monospace">TRANSIENT</text>

        <rect x={444} y={124} width={96} height={28} rx="4" fill="#1a1a1f" stroke="#3f3f46" strokeWidth="1" />
        <text x={492} y={136} textAnchor="middle" fill="#71717a" fontSize="7" fontFamily="monospace" letterSpacing="0.5">HISTORY</text>
        <text x={492} y={146} textAnchor="middle" fill="#22d3ee" fontSize="8" fontFamily="monospace">CAPTURE</text>

        {/* Knob row */}
        <Knob cx={54} cy={196} r={18} value={0.62} color="#a855f7" label="Density" />
        <Knob cx={162} cy={196} r={18} value={0.38} color="#a855f7" label="Grain Len" />
        <Knob cx={270} cy={196} r={18} value={0.5} color="#22d3ee" label="Pitch" />
        <Knob cx={378} cy={196} r={18} value={0.45} color="#22d3ee" label="Feedback" />
        <Knob cx={486} cy={196} r={18} value={0.71} color="#f59e0b" label="Mix" />

        {/* Version tag */}
        <text x={W - 14} y={H - 8} textAnchor="end" fill="#3f3f46" fontSize="7" fontFamily="monospace">
          v0.3-dev
        </text>
      </svg>
    </figure>
  );
}
