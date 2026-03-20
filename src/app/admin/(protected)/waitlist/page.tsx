import { prisma } from "@/lib/prisma";
import { requireAdminPage } from "@/lib/admin";
import { SectionHeader } from "@/components/ui/section-header";

export const dynamic = "force-dynamic";

export default async function AdminWaitlistPage() {
  await requireAdminPage();

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
          <p className="mt-1 text-sm text-muted-foreground">
            {entries.length} signup{entries.length !== 1 ? "s" : ""}
          </p>
        </div>

        {entries.length === 0 ? (
          <div className="rounded-xl border border-border bg-card/50 py-16 text-center">
            <p className="text-muted-foreground">
              No waitlist signups yet. They&apos;ll appear here when someone joins.
            </p>
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-card/50">
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                    Email
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                    Interest
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                    Source
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                    Signed up
                  </th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry, i) => (
                  <tr
                    key={entry.id}
                    className={i % 2 === 0 ? "bg-card/20" : "bg-card/40"}
                  >
                    <td className="px-4 py-3 font-mono text-xs text-foreground">
                      {entry.email}
                    </td>
                    <td className="max-w-xs truncate px-4 py-3 text-muted-foreground">
                      {entry.interest ?? (
                        <span className="italic text-muted-foreground/40">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {entry.source}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
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
