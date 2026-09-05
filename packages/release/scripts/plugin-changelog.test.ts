import { describe, expect, it } from "vitest"
import { reconcilePluginChangelog } from "./plugin-changelog"

describe("plugin release notes", () => {
  const text = "# Pi\n\n## 1.0.1\n\nNew bundled SDK.\n\n## 1.0.0\n\nExisting release.\n"
  it("retitles the prepared entry when an orphaned npm version forces a later patch", () => {
    expect(reconcilePluginChangelog(text, "1.0.1", "1.0.2", true)).toBe(text.replace("## 1.0.1", "## 1.0.2"))
  })
  it("removes only the pending entry when the published plugin is unchanged", () => {
    expect(reconcilePluginChangelog(text, "1.0.1", "1.0.0", false)).toBe("# Pi\n\n## 1.0.0\n\nExisting release.\n")
  })
})
