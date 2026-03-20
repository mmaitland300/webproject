/** Train/eval/review loop for ML case studies (illustrative, not a full pipeline diagram). */
export function MlEvalWorkflowDiagram() {
  return (
    <figure className="my-6 overflow-x-auto rounded-lg border border-border bg-muted/30 p-4">
      <figcaption className="mb-3 text-center text-xs font-medium text-muted-foreground">
        Figure: reproducible evaluation loop (same splits and artifacts every run)
      </figcaption>
      <svg
        viewBox="0 0 640 100"
        className="mx-auto h-auto w-full max-w-2xl text-foreground"
        role="img"
        aria-label="Flow from raw data through clean and split, train, metrics and confusion review, then decide next change"
      >
        <defs>
          <marker
            id="ml-arrow"
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="4"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 Z" className="fill-muted-foreground" />
          </marker>
        </defs>
        {[
          { x: 12, w: 88, label: "Data" },
          { x: 118, w: 92, label: "Clean / split" },
          { x: 226, w: 72, label: "Train" },
          { x: 314, w: 88, label: "Metrics" },
          { x: 418, w: 100, label: "Confusion review" },
          { x: 534, w: 94, label: "Next change" },
        ].map((box, i, arr) => (
          <g key={box.label}>
            <rect
              x={box.x}
              y="28"
              width={box.w}
              height="44"
              rx="6"
              className="fill-card stroke-border"
              strokeWidth="1"
            />
            <text
              x={box.x + box.w / 2}
              y="54"
              textAnchor="middle"
              className="fill-foreground text-[10px] font-medium"
            >
              {box.label}
            </text>
            {i < arr.length - 1 && (
              <line
                x1={box.x + box.w}
                y1="50"
                x2={arr[i + 1].x}
                y2="50"
                className="stroke-muted-foreground"
                strokeWidth="1.5"
                markerEnd="url(#ml-arrow)"
              />
            )}
          </g>
        ))}
      </svg>
    </figure>
  );
}
