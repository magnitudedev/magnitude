import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { ApplicationUpdateState } from "@magnitudedev/sdk/desktop-host"
import { awaitUpdateCheck, renderApplicationUpdate } from "./update-runtime"
const state = (transfer: ApplicationUpdateState["transfer"], check: ApplicationUpdateState["check"] = { _tag: "Succeeded", at: 1 }) => ApplicationUpdateState.make({ transfer, check, preference: { _tag: "Known", autoDownload: false } })
describe("headless application update output", () => {
  it("gives owner-specific installation instructions without suggesting a live server restart", () => {
    const ready = state({ _tag: "Ready", version: "2.0.0" })
    expect(renderApplicationUpdate(ready, "Headless")).toBe("Magnitude 2.0.0 is ready to install.\nStop `magnitude serve`, then run: magnitude update install\n")
    expect(renderApplicationUpdate(ready, "Headless", true)).toBe("Magnitude 2.0.0 is ready to install.\nThe server installs it on its own once it is idle.\n")
    expect(renderApplicationUpdate(ready, "None")).toBe("Magnitude 2.0.0 is ready to install.\nInstall: magnitude update install\n")
    expect(renderApplicationUpdate(ready, "Desktop")).toContain("Install and restart: magnitude update install")
  })
  it("distinguishes available, downloading and prepared updates with exact next commands", () => {
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "2.0.0", bytes: 17800000000, failed: Option.none() }))).toContain("17.8 GB")
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "2.0.0", bytes: 100, failed: Option.none() }))).toContain("magnitude update download")
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "2.0.0", bytes: 100, failed: Option.some("startup") })))
      .toBe("Magnitude 2.0.0 is available (100 B), but it failed to start on this machine, so it is not downloaded automatically.\n"
        + "A newer release replaces it when one is published. To try it again anyway: magnitude update download\n")
    expect(renderApplicationUpdate(state({ _tag: "Downloading", version: "2.0.0", completed: 10, total: 100 }))).toContain("magnitude update status")
    expect(renderApplicationUpdate(state({ _tag: "Ready", version: "2.0.0" }))).toContain("magnitude update install")
  })
  it("never calls an unchecked or failed observation up to date", () => {
    expect(renderApplicationUpdate(state({ _tag: "Idle" }))).toBe("Magnitude is up to date.\n")
    expect(renderApplicationUpdate(state({ _tag: "Idle" }, { _tag: "Idle" }))).toContain("No update check has completed")
    expect(renderApplicationUpdate(state({ _tag: "Idle" }, { _tag: "Failed", message: "Network unavailable" }))).toBe("Network unavailable\n")
    expect(renderApplicationUpdate(state({ _tag: "Unavailable", message: "Install manually" }))).toBe("Install manually\n")
  })
  it("shows a new check instead of the previous download error without hiding an active download", () => {
    expect(renderApplicationUpdate(state({ _tag: "Failed", message: "Previous transfer failed" }, { _tag: "Checking" })))
      .toBe("Checking for application updates.\n")
    expect(renderApplicationUpdate(state({ _tag: "Downloading", version: "2.0.0", completed: 10, total: 100 }, { _tag: "Checking" })))
      .toContain("Downloading Magnitude 2.0.0")
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "1.0.0", bytes: 100, failed: Option.none() }, { _tag: "Checking" })))
      .toBe("Checking for application updates.\n")
  })
  it("waits for a started check to finish before reporting it", async () => {
    const results = [state({ _tag: "Idle" }, { _tag: "Checking" }), state({ _tag: "Idle" })]
    let polls = 0
    const status = Effect.sync(() => ({ state: results[Math.min(polls++, results.length - 1)]! }))
    const finished = await Effect.runPromise(awaitUpdateCheck({ state: state({ _tag: "Idle" }, { _tag: "Checking" }) }, status))
    expect(renderApplicationUpdate(finished.state)).toBe("Magnitude is up to date.\n")
    expect(polls).toBe(2)
  })
})
