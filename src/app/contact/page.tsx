import type { Metadata } from "next";
import { ContactForm } from "@/components/sections/contact-form";
import { Mail, MapPin } from "lucide-react";
import { SectionHeader } from "@/components/ui/section-header";

export const metadata: Metadata = {
  title: "Contact",
  description: "Get in touch — I'd love to hear from you.",
};

export default function ContactPage() {
  return (
    <div className="py-32">
      <div className="mx-auto max-w-4xl px-6">
        <SectionHeader
          eyebrow="Contact"
          title="Let’s Build Something Useful"
          description="Have a question or want to work together? Send me a message and I’ll get back to you as soon as I can."
          className="mb-16"
        />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Info */}
          <div className="space-y-6 lg:col-span-1">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10 shrink-0">
                <Mail className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <h3 className="font-semibold text-sm">Email</h3>
                <p className="text-sm text-muted-foreground">
                  mmaitland300@gmail.com
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/10 shrink-0">
                <MapPin className="h-5 w-5 text-cyan-400" />
              </div>
              <div>
                <h3 className="font-semibold text-sm">Location</h3>
                <p className="text-sm text-muted-foreground">
                  Canon City, Colorado
                </p>
              </div>
            </div>
          </div>

          {/* Form */}
          <div className="lg:col-span-2">
            <ContactForm />
          </div>
        </div>
      </div>
    </div>
  );
}
