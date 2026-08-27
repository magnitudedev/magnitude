import { describe, expect, it } from "vitest"
import {
  isHorizontalSetupStepper,
  isWideSetupLayout,
  setupBodyWidth,
  setupContentWidth,
  setupFrameHeight,
} from "./setup-frame"

describe("setup frame layout", () => {
  it("uses the same bounded content width throughout setup", () => {
    expect(setupContentWidth(160)).toBe(110)
    expect(setupContentWidth(80)).toBe(78)
    expect(setupBodyWidth(160)).toBe(106)
    expect(setupBodyWidth(80)).toBe(74)
  })

  it("uses independent fit thresholds for the model layout and stepper", () => {
    expect(isWideSetupLayout(107)).toBe(true)
    expect(isWideSetupLayout(106)).toBe(false)
    expect(isHorizontalSetupStepper(106)).toBe(true)
    expect(isHorizontalSetupStepper(80)).toBe(true)
    expect(isHorizontalSetupStepper(63)).toBe(true)
    expect(isHorizontalSetupStepper(62)).toBe(false)
  })

  it("uses the model page's required height without the removed title row", () => {
    expect(setupFrameHeight(120)).toBe(26)
    expect(setupFrameHeight(80)).toBe(44)
    expect(setupFrameHeight(60)).toBe(48)
    expect(setupFrameHeight(120, 1)).toBe(27)
  })
})
