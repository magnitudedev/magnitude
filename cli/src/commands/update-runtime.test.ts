import { describe, expect, it } from "vitest"
import { DesktopUpdateState } from "@magnitudedev/sdk/desktop-host"
import { renderApplicationUpdate } from "./update-runtime"
const state = (transfer: DesktopUpdateState["transfer"], check: DesktopUpdateState["check"] = { _tag: "Succeeded", at: 1 }) => DesktopUpdateState.make({ transfer, check, preference: { _tag: "Known", autoDownload: false } })
describe("headless application update output", () => {
  it("distinguishes available, downloading and prepared updates with exact next commands", () => {
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "2.0.0", bytes: 17800000000 }))).toContain("17.8 GB")
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "2.0.0", bytes: 100 }))).toContain("magnitude update download")
    expect(renderApplicationUpdate(state({ _tag: "Downloading", version: "2.0.0", completed: 10, total: 100 }))).toContain("magnitude update status")
    expect(renderApplicationUpdate(state({ _tag: "Ready", version: "2.0.0" }))).toContain("magnitude update install")
  })
  it("never calls an unchecked or failed observation up to date", () => {
    expect(renderApplicationUpdate(state({ _tag: "Idle" }))).toBe("Magnitude is up to date.\n")
    expect(renderApplicationUpdate(state({ _tag: "Idle" }, { _tag: "Idle" }))).toContain("No update check has completed")
    expect(renderApplicationUpdate(state({ _tag: "Idle" }, { _tag: "Failed", message: "Network unavailable" }))).toBe("Network unavailable\n")
    expect(renderApplicationUpdate(state({ _tag: "Unavailable", message: "Install manually" }))).toBe("Install manually\n")
  })
})
