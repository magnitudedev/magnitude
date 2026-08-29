import { describe, expect, it } from "vitest"
import {
  SPINNER_FRAME_MS,
  spinnerFrameAt,
  spinnerFrameForStep,
} from "./spinner"

describe("spinner", () => {
  it("advances from elapsed time at one shared frame rate", () => {
    expect(spinnerFrameAt(0)).toBe(spinnerFrameForStep(0))
    expect(spinnerFrameAt(SPINNER_FRAME_MS - 1)).toBe(spinnerFrameForStep(0))
    expect(spinnerFrameAt(SPINNER_FRAME_MS)).toBe(spinnerFrameForStep(1))
  })
})
