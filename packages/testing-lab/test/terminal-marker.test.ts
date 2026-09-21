import { expect, test } from "vitest"
import { containsTerminalMarker, screenContainsTerminalMarker } from "../src/harnesses/terminal"

test("generated terminal markers survive renderer word wrapping", () => {
  // Native Hermes evidence wrapped this word inside its reasoning panel at 80 columns.
  expect(screenContainsTerminalMarker(['1. Concatenate "SUN" and "FLOWER" without a space -> "SUNFLOW', 'ER"'], "SUNFLOWER")).toBe(true)
  expect(screenContainsTerminalMarker(["sunflow", "er"], "SUNFLOWER")).toBe(true)
  expect(screenContainsTerminalMarker(["First concatenate SUN and FLOWER without a space."], "SUNFLOWER")).toBe(false)
  expect(screenContainsTerminalMarker(["First concatenate SUN and", "FLOWER without a space."], "SUNFLOWER")).toBe(false)
  expect(screenContainsTerminalMarker(["SUN FLOWER"], "SUNFLOWER")).toBe(false)
  expect(screenContainsTerminalMarker(["anything"], "")).toBe(false)
  // Transcript comparison remains exact apart from case, independent of display wrapping.
  expect(containsTerminalMarker("SUN\nFLOWER", "SUNFLOWER")).toBe(false)
})
