import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { submitBlocked } from "./submit-blocked"

describe("a send or retry that can't go", () => {
  it("says the chat's model is gone and to choose another, rather than doing nothing", () => {
    const reason = submitBlocked({ chosen: true, available: false, settingsValid: true, servable: true })
    expect(Option.getOrThrow(reason)).toContain("isn’t available anymore")
    expect(Option.getOrThrow(reason)).toContain("Choose another model")
  })

  it("says to choose a model when none is chosen, and to fix its settings when they are out of range", () => {
    expect(Option.getOrThrow(submitBlocked({ chosen: false, available: false, settingsValid: true, servable: true }))).toContain("Choose a model")
    expect(Option.getOrThrow(submitBlocked({ chosen: true, available: true, settingsValid: false, servable: true }))).toContain("Settings → Chat")
  })

  it("says to download a model, not to choose another, when no other model is there to choose", () => {
    const gone = Option.getOrThrow(submitBlocked({ chosen: true, available: false, settingsValid: true, servable: false }))
    expect(gone).toContain("no other model is downloaded")
    expect(gone).not.toContain("beside Send")
    expect(Option.getOrThrow(submitBlocked({ chosen: false, available: false, settingsValid: true, servable: false }))).toBe("Download a model in Models, then try again.")
  })

  it("says nothing when the model is ready and its settings are valid", () => {
    expect(submitBlocked({ chosen: true, available: true, settingsValid: true, servable: true })).toEqual(Option.none())
  })
})
