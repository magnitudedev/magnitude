import { Effect, Option } from "effect"
import { describe, expect, it, vi } from "vitest"
import { ApplicationUpdateState } from "@magnitudedev/sdk/desktop-host"
import { awaitUpdateCheck, renderApplicationUpdate, renderIdleInstallation, updateCommand, updateFailureMessage } from "./update-runtime"
const state = (transfer: ApplicationUpdateState["transfer"], check: ApplicationUpdateState["check"] = { _tag: "Succeeded", at: 1 }) => ApplicationUpdateState.make({ transfer, check, preference: { _tag: "Known", autoDownload: false } })
describe("headless application update output", () => {
  it("gives owner-specific installation instructions without suggesting a live server restart", () => {
    const ready = state({ _tag: "Ready", version: "2.0.0" })
    expect(renderApplicationUpdate(ready, "Headless")).toBe("Magnitude 2.0.0 is ready to install.\n`magnitude serve` installs it once it is idle.\n")
    expect(renderApplicationUpdate(ready, "None")).toBe("Magnitude 2.0.0 is ready to install.\nInstall: magnitude update install\n")
    expect(renderApplicationUpdate(ready, "Desktop")).toContain("Install and restart: magnitude update install")
  })
  it("says a running serve installs an accepted update once idle, and otherwise reports why nothing installs", () => {
    expect(renderIdleInstallation(state({ _tag: "Downloading", version: "2.0.0", completed: 1, total: 2 })))
      .toBe("Magnitude 2.0.0 is downloading. Once it is verified, `magnitude serve` installs it when idle and restarts into the new version.\nCheck progress: magnitude update status\n")
    expect(renderIdleInstallation(state({ _tag: "Ready", version: "2.0.0" })))
      .toBe("Magnitude 2.0.0 will be installed once `magnitude serve` is idle; it restarts into the new version.\nCheck progress: magnitude update status\n")
    expect(renderIdleInstallation(state({ _tag: "Idle" }))).toBe("Magnitude is up to date.\n")
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
  it("recommends `magnitude update install` for available updates on CLI-only installs", () => {
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "2.0.0", bytes: 100, failed: Option.none() }), "Headless", true))
      .toBe("Magnitude 2.0.0 is available (100 B).\nInstall: magnitude update install\n")
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "2.0.0", bytes: 100, failed: Option.some("startup") }), "None", true))
      .toBe("Magnitude 2.0.0 is available (100 B), but it failed to start on this machine, so it is not downloaded automatically.\n"
        + "A newer release replaces it when one is published. To try it again anyway: magnitude update install\n")
  })
  it("tells a container to pull the newer image instead of downloading", () => {
    expect(renderApplicationUpdate(state({ _tag: "ImageAvailable", version: "2.0.0", image: "magnitudedev/magnitude" }), "Headless"))
      .toBe("Magnitude 2.0.0 is available. Pull the new image (docker pull magnitudedev/magnitude) and recreate the container.\n")
    expect(renderApplicationUpdate(state({ _tag: "ImageAvailable", version: "2.0.0", image: "magnitudedev/magnitude" }, { _tag: "Checking" }))).toBe("Checking for updates.\n")
  })
  it("never calls an unchecked or failed observation up to date", () => {
    expect(renderApplicationUpdate(state({ _tag: "Idle" }))).toBe("Magnitude is up to date.\n")
    expect(renderApplicationUpdate(state({ _tag: "Idle" }, { _tag: "Idle" }))).toContain("No update check has completed")
    expect(renderApplicationUpdate(state({ _tag: "Idle" }, { _tag: "Failed", message: "Network unavailable" }))).toBe("Network unavailable\n")
    expect(renderApplicationUpdate(state({ _tag: "Unavailable", kind: "NotInThisBuild", message: "Install manually" }))).toBe("Install manually\n")
  })
  it("shows a new check instead of the previous download error without hiding an active download", () => {
    expect(renderApplicationUpdate(state({ _tag: "Failed", message: "Previous transfer failed" }, { _tag: "Checking" })))
      .toBe("Checking for updates.\n")
    expect(renderApplicationUpdate(state({ _tag: "Downloading", version: "2.0.0", completed: 10, total: 100 }, { _tag: "Checking" })))
      .toContain("Downloading Magnitude 2.0.0")
    expect(renderApplicationUpdate(state({ _tag: "Available", version: "1.0.0", bytes: 100, failed: Option.none() }, { _tag: "Checking" })))
      .toBe("Checking for updates.\n")
  })
  it("waits for a started check to finish before reporting it", async () => {
    const results = [state({ _tag: "Idle" }, { _tag: "Checking" }), state({ _tag: "Idle" })]
    let polls = 0
    const status = Effect.sync(() => ({ state: results[Math.min(polls++, results.length - 1)]! }))
    const finished = await Effect.runPromise(awaitUpdateCheck({ state: state({ _tag: "Idle" }, { _tag: "Checking" }) }, status))
    expect(renderApplicationUpdate(finished.state)).toBe("Magnitude is up to date.\n")
    expect(polls).toBe(2)
  })
  it("exits non-zero and writes to stderr when `update check` ends in a failed check or transfer", async () => {
    expect(updateFailureMessage("check", state({ _tag: "Idle" }, { _tag: "Failed", message: "Network unavailable" })))
      .toEqual(Option.some("Network unavailable"))
    expect(updateFailureMessage("check", state({ _tag: "Failed", message: "Download failed" })))
      .toEqual(Option.some("Download failed"))
    expect(updateFailureMessage("status", state({ _tag: "Idle" }, { _tag: "Failed", message: "Network unavailable" })))
      .toEqual(Option.none())

    const stdoutChunks: string[] = []
    const stderrChunks: string[] = []
    const stdoutSpy = vi.spyOn(process.stdout, "write").mockImplementation((chunk) => { stdoutChunks.push(String(chunk)); return true })
    const stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation((chunk) => { stderrChunks.push(String(chunk)); return true })
    const previousExitCode = process.exitCode
    try {
      process.exitCode = 0
      await Effect.runPromise(updateCommand("check", {
        isStandalone: Effect.succeed(false),
        runAction: () => Effect.succeed({
          state: state({ _tag: "Idle" }, { _tag: "Failed", message: "Update server unreachable" }),
          owner: "Headless",
        }),
      }))
      expect(process.exitCode).toBe(1)
      expect(stderrChunks.join("")).toBe("Update server unreachable\n")
      expect(stdoutChunks.join("")).toBe("")

      stdoutChunks.length = 0
      stderrChunks.length = 0
      process.exitCode = 0
      await Effect.runPromise(updateCommand("check", {
        isStandalone: Effect.succeed(false),
        runAction: () => Effect.succeed({
          state: state({ _tag: "Idle" }, { _tag: "Succeeded", at: 1 }),
          owner: "Headless",
        }),
      }))
      expect(process.exitCode).toBe(0)
      expect(stdoutChunks.join("")).toBe("Magnitude is up to date.\n")
      expect(stderrChunks.join("")).toBe("")
    } finally {
      process.exitCode = previousExitCode
      stdoutSpy.mockRestore()
      stderrSpy.mockRestore()
    }
  })
})
