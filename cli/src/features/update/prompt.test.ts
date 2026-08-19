import { describe, expect, it } from "vitest"
import { adjacentUpdateSelection } from "./prompt"

describe("update prompt selection", () => {
  it("cycles through the three Codex-style choices", () => {
    expect(adjacentUpdateSelection("Update", 1)).toBe("Skip")
    expect(adjacentUpdateSelection("Skip", 1)).toBe("Dismiss")
    expect(adjacentUpdateSelection("Dismiss", 1)).toBe("Update")
    expect(adjacentUpdateSelection("Update", -1)).toBe("Dismiss")
  })
})
