import { describe, expect, it } from "vitest"
import { isWideSetupLayout, setupBodyWidth, setupContentWidth, setupFrameHeight } from "./setup-frame"

describe("setup frame layout", () => {
  it("uses the same bounded content width throughout setup", () => {
    expect(setupContentWidth(160)).toBe(110)
    expect(setupContentWidth(80)).toBe(78)
    expect(setupBodyWidth(160)).toBe(106)
    expect(setupBodyWidth(80)).toBe(74)
  })

  it("switches the stepper at the model chooser layout breakpoint", () => {
    expect(isWideSetupLayout(107)).toBe(true)
    expect(isWideSetupLayout(106)).toBe(false)
  })

  it("uses the model page's required height without the removed title row", () => {
    expect(setupFrameHeight(120)).toBe(26)
    expect(setupFrameHeight(80)).toBe(48)
    expect(setupFrameHeight(120, 1)).toBe(27)
  })
})
