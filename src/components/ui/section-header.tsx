import { cn } from "@/lib/utils";

interface SectionHeaderProps {
  title: string;
  description?: string;
  eyebrow?: string;
  align?: "left" | "center";
  titleClassName?: string;
  className?: string;
}

export function SectionHeader({
  title,
  description,
  eyebrow,
  align = "center",
  titleClassName,
  className,
}: SectionHeaderProps) {
  const centered = align === "center";

  return (
    <div
      className={cn(
        "space-y-4",
        centered ? "text-center" : "text-left",
        className
      )}
    >
      {eyebrow && (
        <div
          className={cn(
            "flex items-center gap-3",
            centered ? "justify-center" : "justify-start"
          )}
        >
          <span className="h-px w-10 bg-[linear-gradient(90deg,rgba(122,162,247,0.9),rgba(122,162,247,0.15))]" />
          <p className="section-eyebrow">{eyebrow}</p>
        </div>
      )}
      <h1
        className={cn(
          "page-title cyber-title text-balance text-3xl font-semibold tracking-tight sm:text-4xl",
          titleClassName
        )}
      >
        {title}
      </h1>
      {description && (
        <p
          className={cn(
            "text-pretty text-muted-foreground",
            centered ? "mx-auto max-w-2xl" : "max-w-2xl"
          )}
        >
          {description}
        </p>
      )}
    </div>
  );
}
