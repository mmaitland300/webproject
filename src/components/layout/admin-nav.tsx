"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const adminLinks = [
  { href: "/admin/inbox", label: "Inbox" },
  { href: "/admin/waitlist", label: "Waitlist" },
  { href: "/admin/comments", label: "Comments" },
];

export function AdminNav() {
  const pathname = usePathname();

  return (
    <div className="sticky top-16 z-40 border-b border-border bg-background/85 backdrop-blur-sm">
      <div className="mx-auto flex max-w-4xl gap-4 px-6 py-3 text-sm">
        {adminLinks.map((link) => {
          const isActive =
            pathname === link.href || pathname.startsWith(`${link.href}/`);

          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "rounded-md px-2 py-1 transition-colors",
                isActive
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
              aria-current={isActive ? "page" : undefined}
            >
              {link.label}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
