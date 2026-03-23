import { createOgImage, ogSize } from "@/lib/og-image";

export const runtime = "edge";
export const alt = "Contact | Matt Maitland";
export const size = ogSize;
export const contentType = "image/png";

export default function OgImage() {
  return createOgImage("Contact", "Get in touch — project inquiries, collaboration, or just to say hello");
}
