export default function RootLoading() {
  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-6">
      <div className="w-full max-w-xl rounded-xl border border-border bg-card/50 p-6">
        <div className="h-5 w-40 animate-pulse rounded bg-muted" />
        <div className="mt-4 space-y-2">
          <div className="h-3 w-full animate-pulse rounded bg-muted" />
          <div className="h-3 w-[92%] animate-pulse rounded bg-muted" />
          <div className="h-3 w-[86%] animate-pulse rounded bg-muted" />
        </div>
      </div>
    </div>
  );
}
