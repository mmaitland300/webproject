import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Matt Maitland | Developer";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OgImage() {
  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#0a0a0a",
          fontFamily: "sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 20,
          }}
        >
          <div
            style={{
              fontSize: 72,
              fontWeight: 800,
              background: "linear-gradient(135deg, #8b5cf6, #06b6d4)",
              backgroundClip: "text",
              color: "transparent",
            }}
          >
            Matt Maitland
          </div>
          <div
            style={{
              fontSize: 28,
              color: "#a1a1aa",
            }}
          >
            Full-Stack Developer
          </div>
        </div>
      </div>
    ),
    { ...size }
  );
}
