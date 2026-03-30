export default function BlogLoading() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mb-12 space-y-3">
          <div className="h-3 w-20 animate-pulse rounded bg-muted" />
          <div className="h-8 w-64 animate-pulse rounded bg-muted" />
          <div className="h-3 w-[24rem] max-w-full animate-pulse rounded bg-muted" />
        </div>
        <div className="space-y-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="rounded-xl border border-border bg-card/50 p-5 space-y-3"
            >
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
