import { describe, expect, it } from "vitest"
import { installMethodFromEnvironment } from "./install-context"
import { updateActionFor, updateCommandString } from "./package-manager-actions"

describe("package-manager update actions", () => {
  it("maps launcher provenance to version-pinned global update commands", () => {
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY: "npm" }))
      .toBe("npm")
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY: "bun" }))
      .toBe("bun")
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY: "pnpm" }))
      .toBe("pnpm")
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY: "yarn" }))
      .toBe("other")
    expect(installMethodFromEnvironment({})).toBe("other")

    expect(updateCommandString(updateActionFor("npm", "1.2.3")))
      .toBe("npm install -g @magnitudedev/cli@1.2.3")
    expect(updateCommandString(updateActionFor("bun", "1.2.3")))
      .toBe("bun install -g @magnitudedev/cli@1.2.3")
    expect(updateCommandString(updateActionFor("pnpm", "1.2.3")))
      .toBe("pnpm add -g @magnitudedev/cli@1.2.3")
  })
})
