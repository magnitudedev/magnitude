import { describe, expect, it } from "vitest"
import type { DisplayRootStatus } from "@magnitudedev/sdk"
import { isWorkStatusBarVisible } from "./work-status-bar"

const worked: DisplayRootStatus = {
  _tag: "Worked",
  lastProductiveMs: 5_000,
}

const working: DisplayRootStatus = {
  _tag: "Working",
  chainStartedAt: 1,
  detail: { _tag: "NoDetail" },
  activeChildCount: 0,
}

describe("isWorkStatusBarVisible", () => {
  it("shows live work above the composer", () => {
    expect(isWorkStatusBarVisible(working, false)).toBe(true)
  })

  it("moves completed-work presentation into the timeline", () => {
    expect(isWorkStatusBarVisible(worked, false)).toBe(false)
  })

  it("keeps the task panel available after root work completes", () => {
    expect(isWorkStatusBarVisible(worked, true)).toBe(true)
  })
})
