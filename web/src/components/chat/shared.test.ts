import { describe, expect, it } from "vitest"
import { duration, sentAt, thinkingLabel } from "./shared"

describe("how long something ran", () => {
  it("counts seconds, then minutes, then hours", () => {
    expect([0, 15_000, 45_900, 60_000, 75_000, 3_599_000, 7_500_000].map(duration)).toEqual(["0s", "15s", "45s", "1m 0s", "1m 15s", "59m 59s", "2h 5m"])
  })
  it("says Thinking for its first second, then for how long", () => {
    expect([thinkingLabel(400), thinkingLabel(4_000), thinkingLabel(125_000)]).toEqual(["Thinking", "Thinking for 4s", "Thinking for 2m 5s"])
  })
})

const now = new Date(2026, 10, 18, 15, 18).getTime()

describe("when a message was sent", () => {
  it("says how long ago within a day, then the date and time", () => {
    expect(sentAt(now - 20_000, now)).toBe("just now")
    expect(sentAt(now - 60_000, now)).toBe("1m ago")
    expect(sentAt(now - 59 * 60_000, now)).toBe("59m ago")
    expect(sentAt(now - 2 * 3_600_000, now)).toBe("2hr ago")
    expect(sentAt(now - 23 * 3_600_000, now)).toBe("23hr ago")
    const older = sentAt(new Date(2026, 10, 16, 15, 18).getTime(), now)
    expect(older).toMatch(/Nov 16, 3:18pm/)
    expect(sentAt(new Date(2025, 10, 16, 9, 5).getTime(), now)).toMatch(/2025/)
  })
})
