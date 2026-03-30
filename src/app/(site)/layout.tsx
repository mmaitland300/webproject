import { SiteChrome } from "@/components/layout/site-chrome";

/**
 * Public marketing pages: global nav, footer, dark theme shell, decorative bg.
 * Routes outside this group (for example /resume/print) stay minimal for PDF and print.
 */
export default function SiteLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <SiteChrome>{children}</SiteChrome>;
}
