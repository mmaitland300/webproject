import { requireAdminPage } from "@/lib/admin";
import { SectionHeader } from "@/components/ui/section-header";
import { AdminComposeForm } from "@/components/sections/admin-compose-form";

export const dynamic = "force-dynamic";

export default async function AdminComposePage() {
  await requireAdminPage();

  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <SectionHeader
          align="left"
          eyebrow="Admin"
          title="Compose Email"
          className="mb-8 space-y-2"
          titleClassName="text-3xl"
        />
        <AdminComposeForm />
      </div>
    </div>
  );
}
