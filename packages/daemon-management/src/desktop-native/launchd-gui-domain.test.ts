import { describe, expect, it } from "vitest"
import { launchdGuiDomainIsAbsent } from "./launchd-gui-domain"

describe("launchd GUI domain absence", () => {
  it("recognizes both forms launchd reports for a user without a GUI session", () => {
    expect(launchdGuiDomainIsAbsent({ code: 112, stderr: "Bad request.\nCould not find domain for user gui: 501\n" }, 501)).toBe(true)
    expect(launchdGuiDomainIsAbsent({ code: 125, stderr: "Could not print domain: 125: Domain does not support specified action\n" }, 501)).toBe(true)
  })
  it("keeps every other lookup failure distinct", () => {
    for (const result of [
      { code: 112, stderr: "Bad request.\nCould not find domain for user gui: 502" },
      { code: 112, stderr: "Operation not permitted" },
      { code: 125, stderr: "Operation not permitted" },
      { code: 1, stderr: "Could not print domain: 125: Domain does not support specified action" },
      { code: 113, stderr: "Could not find service" },
    ]) expect(launchdGuiDomainIsAbsent(result, 501)).toBe(false)
  })
})
