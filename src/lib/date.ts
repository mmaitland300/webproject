const DATE_ONLY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

export function parseDateValue(value: string | Date): Date {
  if (value instanceof Date) {
    return value;
  }

  if (DATE_ONLY_PATTERN.test(value)) {
    return new Date(`${value}T00:00:00.000Z`);
  }

  return new Date(value);
}

export function formatDisplayDate(
  value: string | Date,
  locale = "en-US",
  options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "long",
    day: "numeric",
  }
): string {
  const isDateOnly = typeof value === "string" && DATE_ONLY_PATTERN.test(value);

  return new Intl.DateTimeFormat(locale, {
    ...options,
    ...(isDateOnly ? { timeZone: "UTC" } : {}),
  }).format(parseDateValue(value));
}
