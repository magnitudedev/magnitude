/**
 * Reminder composition helper.
 *
 * Combines multiple nullable reminder parts into a single string
 * for injection via TurnResult.reminder.
 */

/**
 * Combine multiple reminder parts (some possibly null) into a single string.
 * Returns null if all parts are null/empty.
 */
export function buildReminder(...parts: (string | null)[]): string | null {
  const filtered = parts.filter((p): p is string => p !== null)
  return filtered.length ? filtered.join('\n\n') : null
}
