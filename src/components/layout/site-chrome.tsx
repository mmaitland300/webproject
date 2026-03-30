import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";

/** Dark portfolio shell: nav, footer, decorative background. */
export function SiteChrome({ children }: { children: React.ReactNode }) {
  return (
    <div className="dark min-h-full flex flex-col relative">
      <div className="animated-bg-glow" aria-hidden="true" />
      <Navbar />
      <main className="flex-1 relative z-10">{children}</main>
      <Footer />
    </div>
  );
}
