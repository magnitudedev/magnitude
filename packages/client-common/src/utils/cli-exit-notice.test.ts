import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { CLI_EXIT_OBSERVATION_FALLBACK, deriveCliExitNotice } from "./cli-exit-notice"

describe("deriveCliExitNotice", () => {
  it("does not warn after a healthy background-service close", () => {
    expect(Option.getOrUndefined(deriveCliExitNotice(Option.some({ connectedClientCount: 0 })))).toBeUndefined()
  })

  it("uses bounded fallback copy when close observation failed", () => {
    expect(Option.getOrUndefined(deriveCliExitNotice(Option.none()))).toBe(CLI_EXIT_OBSERVATION_FALLBACK)
  })
})
