import { describe, expect, it } from "vitest"
import { formatMessageRelativeTime } from "./message-relative-time"

const MINUTE = 60_000
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

describe("formatMessageRelativeTime", () => {
  const now = 10 * 365 * DAY

  it.each([
    [now + MINUTE, "Just now"],
    [now - 59_999, "Just now"],
    [now - MINUTE, "1 minute ago"],
    [now - (15 * MINUTE), "15 minutes ago"],
    [now - HOUR, "1 hour ago"],
    [now - (5 * HOUR), "5 hours ago"],
    [now - DAY, "1 day ago"],
    [now - (6 * DAY), "6 days ago"],
    [now - (14 * DAY), "2 weeks ago"],
    [now - (60 * DAY), "2 months ago"],
    [now - (2 * 365 * DAY), "2 years ago"],
  ])("formats %s as %s", (timestamp, expected) => {
    expect(formatMessageRelativeTime(timestamp, now)).toBe(expected)
  })
})
