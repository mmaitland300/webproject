/** Public Snake Detector demo base URL; optional until a hosted demo exists. */
export function getSnakeDemoUrl(): string | undefined {
  const raw = process.env.NEXT_PUBLIC_SNAKE_DEMO_URL?.trim();
  if (!raw) return undefined;
  if (!/^https?:\/\//i.test(raw)) return undefined;
  return raw;
}
