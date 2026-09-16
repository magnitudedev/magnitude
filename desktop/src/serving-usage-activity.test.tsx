import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { Schema } from "effect"
import { ServingUsageSnapshot } from "@magnitudedev/sdk"
import { activityLevel, activitySummary, UsageActivity } from "./serving-usage-activity"

const days = (tokens: number[]) => Schema.decodeUnknownSync(ServingUsageSnapshot.members[1].fields.dailyActivity)(tokens.map((totalTokens, index) => ({ date: new Date(Date.UTC(2024, 1, 25 + index)).toISOString().slice(0, 10), totalTokens })))
it("keeps yesterday's streak alive today and breaks it after a missed day", () => {
  expect(activitySummary(days([2, 3, 4, 0, 5, 6, 0]))).toEqual({ totalTokens: 20, currentStreak: 2 })
  expect(activitySummary(days([2, 3, 0, 0])).currentStreak).toBe(0)
  expect(activitySummary(days([2, 3, 4])).currentStreak).toBe(3)
  expect(activitySummary(days([0, 0, 0]))).toEqual({ totalTokens: 0, currentStreak: 0 })
})
it("uses a zero level and four bounded nonzero blue levels", () => {
  expect([0, 1, 25, 26, 50, 51, 75, 76, 100].map(value => activityLevel(value, 100))).toEqual([0, 1, 1, 2, 2, 3, 3, 4, 4])
})
it("exposes exact dates and tokens, one tab stop, and no future days", () => {
  const html = renderToStaticMarkup(<UsageActivity days={days([1, 2, 3, 4, 5, 6])} />)
  expect(html).toContain('5 tokens on February 29, 2024')
  expect(html.match(/tabindex="0"/g)).toHaveLength(1)
  expect(html.match(/data-day-index=/g)).toHaveLength(6)
  expect(html).toContain('bg-blue-700')
})
it("keeps unobserved totals and streaks unknown", () => {
  const html = renderToStaticMarkup(<UsageActivity days={null} />)
  expect(html).toContain('aria-busy="true"')
  expect(html).not.toContain('0 tokens')
  expect(html).not.toContain('<button')
})
