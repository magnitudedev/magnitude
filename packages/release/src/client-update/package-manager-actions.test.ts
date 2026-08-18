import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { installMethodFromEnvironment } from "./install-context"
import { updateActionFor, updateCommandString } from "./package-manager-actions"

describe("package-manager update actions", () => {
  it("maps launcher provenance to Codex-style global update commands", () => {
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY: "npm" }))
      .toBe("npm")
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY: "bun" }))
      .toBe("bun")
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY: "pnpm" }))
      .toBe("pnpm")
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY: "yarn" }))
      .toBe("other")
    expect(installMethodFromEnvironment({})).toBe("other")

    expect(updateCommandString(Option.getOrThrow(updateActionFor("npm"))))
      .toBe("npm install -g @magnitudedev/cli")
    expect(updateCommandString(Option.getOrThrow(updateActionFor("bun"))))
      .toBe("bun install -g @magnitudedev/cli")
    expect(updateCommandString(Option.getOrThrow(updateActionFor("pnpm"))))
      .toBe("pnpm add -g @magnitudedev/cli")
    expect(Option.isNone(updateActionFor("other"))).toBe(true)
  })
})
