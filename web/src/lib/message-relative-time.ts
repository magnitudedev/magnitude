const SECOND_MS = 1_000
const MINUTE_MS = 60 * SECOND_MS
const HOUR_MS = 60 * MINUTE_MS
const DAY_MS = 24 * HOUR_MS
const WEEK_MS = 7 * DAY_MS
const MONTH_MS = 30 * DAY_MS
const YEAR_MS = 365 * DAY_MS

function ago(value: number, unit: string): string {
  return `${value} ${unit}${value === 1 ? "" : "s"} ago`
}

/** Long relative time for message metadata. Future clock skew is treated as now. */
export function formatMessageRelativeTime(timestamp: number, now: number): string {
  const elapsed = Math.max(0, now - timestamp)
  if (elapsed < MINUTE_MS) return "Just now"
  if (elapsed < HOUR_MS) return ago(Math.floor(elapsed / MINUTE_MS), "minute")
  if (elapsed < DAY_MS) return ago(Math.floor(elapsed / HOUR_MS), "hour")
  if (elapsed < WEEK_MS) return ago(Math.floor(elapsed / DAY_MS), "day")
  if (elapsed < MONTH_MS) return ago(Math.floor(elapsed / WEEK_MS), "week")
  if (elapsed < YEAR_MS) return ago(Math.floor(elapsed / MONTH_MS), "month")
  return ago(Math.floor(elapsed / YEAR_MS), "year")
}
