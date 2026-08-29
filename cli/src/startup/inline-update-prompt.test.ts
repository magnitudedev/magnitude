import { describe, expect, it } from "vitest"
import { updateActionFor } from "@magnitudedev/release"
import { defaultCliThemes } from "../utils/theme"
import {
  adjacentInlineUpdateSelection,
  renderInlineUpdatePrompt,
} from "./inline-update-prompt"

describe("inline update prompt", () => {
  it("renders the agreed numbered selector without extra control hints", () => {
    expect(renderInlineUpdatePrompt(
      "1.3.0",
      "1.4.0",
      updateActionFor("npm", "1.4.0"),
      "Update",
      defaultCliThemes.dark,
      false,
    )).toEqual([
      "Update available! 1.3.0 → 1.4.0",
      "",
      "Release notes: https://github.com/magnitudedev/magnitude/releases/tag/@magnitudedev/cli@1.4.0",
      "",
      "› 1. Update now (runs `npm install -g @magnitudedev/cli@1.4.0`)",
      "  2. Skip",
      "  3. Skip until next version",
      "",
      "Press enter to continue",
    ])
  })

  it("wraps arrow navigation", () => {
    expect(adjacentInlineUpdateSelection("Update", -1)).toBe("Dismiss")
    expect(adjacentInlineUpdateSelection("Dismiss", 1)).toBe("Update")
  })
})
