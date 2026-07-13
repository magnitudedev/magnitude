/**
 * Format a size in tokens as a human-readable string.
 * e.g., 4000 -> "4k", 500 -> "500"
 */
export function formatSize(tokens: number): string {
  if (tokens < 1000) return String(tokens)
  const k = tokens / 1000
  return k >= 10 ? `${Math.round(k)}k` : `${k.toFixed(1)}k`
}
