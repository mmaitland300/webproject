import { prisma } from "@/lib/prisma";
import { SectionHeader } from "@/components/ui/section-header";

export const dynamic = "force-dynamic";

export default async function AdminWaitlistPage() {
  const entries = await prisma.stringFluxWaitlist.findMany({
    orderBy: { createdAt: "desc" },
  });

  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <div className="mb-8">
          <SectionHeader
            align="left"
            eyebrow="Admin"
            title="StringFlux Waitlist"
            className="space-y-2"
            titleClassName="text-3xl"
          />
          <p className="text-sm text-muted-foreground mt-1">
            {entries.length} signup{entries.length !== 1 ? "s" : ""}
          </p>
        </div>

        {entries.length === 0 ? (
          <div className="text-center py-16 border border-border rounded-xl bg-card/50">
            <p className="text-muted-foreground">
              No waitlist signups yet. They&apos;ll appear here when someone joins.
            </p>
          </div>
        ) : (
          <div className="rounded-xl border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-card/50">
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Email</th>
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Interest</th>
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Source</th>
                  <th className="text-left px-4 py-3 font-medium text-muted-foreground">Signed up</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry, i) => (
                  <tr
                    key={entry.id}
                    className={
                      i % 2 === 0 ? "bg-card/20" : "bg-card/40"
                    }
                  >
                    <td className="px-4 py-3 font-mono text-xs text-foreground">
                      {entry.email}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground max-w-xs truncate">
                      {entry.interest ?? <span className="text-muted-foreground/40 italic">—</span>}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {entry.source}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground whitespace-nowrap">
                      {entry.createdAt.toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
