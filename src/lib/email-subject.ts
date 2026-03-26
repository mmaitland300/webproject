const RE_PREFIX_PATTERN = /^\s*(?:re:\s*)+/i;

export function normalizeReplySubject(input: string): string {
  const stripped = input.replace(RE_PREFIX_PATTERN, "").trim();
  const fallback = stripped.length > 0 ? stripped : "Your message";
  return `Re: ${fallback}`;
}
