export default function ProjectsLoading() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mb-12 space-y-3">
          <div className="h-3 w-24 animate-pulse rounded bg-muted" />
          <div className="h-8 w-72 animate-pulse rounded bg-muted" />
          <div className="h-3 w-[26rem] max-w-full animate-pulse rounded bg-muted" />
        </div>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="rounded-xl border border-border bg-card/50 p-5 space-y-3"
            >
              <div className="h-40 w-full animate-pulse rounded-md bg-muted" />
              <div className="h-5 w-2/3 animate-pulse rounded bg-muted" />
              <div className="h-3 w-full animate-pulse rounded bg-muted" />
              <div className="h-3 w-5/6 animate-pulse rounded bg-muted" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
