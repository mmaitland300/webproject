import { createOgImage, ogSize } from "@/lib/og-image";

export const runtime = "edge";
export const alt = "Projects | Matt Maitland";
export const size = ogSize;
export const contentType = "image/png";

export default function OgImage() {
  return createOgImage("Projects", "Web apps, audio plugins, ML experiments, and case studies");
}
