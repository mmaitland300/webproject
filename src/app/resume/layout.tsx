"use client";

import { usePathname } from "next/navigation";
import { SiteChrome } from "@/components/layout/site-chrome";

/**
 * /resume uses portfolio chrome; /resume/print stays bare for PDF and paper layout.
 * (Single app/resume tree avoids Next.js route-group merge edge cases for /resume/print.)
 */
export default function ResumeShellLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  if (pathname === "/resume/print") {
    return <>{children}</>;
  }
  return <SiteChrome>{children}</SiteChrome>;
}
