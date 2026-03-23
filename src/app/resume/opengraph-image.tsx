import { createOgImage, ogSize } from "@/lib/og-image";

export const runtime = "edge";
export const alt = "Resume | Matt Maitland";
export const size = ogSize;
export const contentType = "image/png";

export default function OgImage() {
  return createOgImage("Resume", "Experience, skills, and education");
}
