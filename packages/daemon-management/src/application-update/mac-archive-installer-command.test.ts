import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { decodeMacArchiveInstallerRequest } from "./mac-archive-installer-command"

const request = { bundle: "/Applications/Magnitude.app", archive: "/private/tmp/download/magnitude.zip",
  channel: "stable", offer: { download: "https://github.com/magnitudedev/magnitude/releases/download/test/magnitude.zip",
    release: { version: "0.1.6", bytes: 1, sha256: "0".repeat(64), signature: "A".repeat(86) + "==" } } }

describe("macOS archive installer input", () => {
  it("accepts a bounded request without authorizing its release signature", async () => {
    expect(await Effect.runPromise(decodeMacArchiveInstallerRequest(JSON.stringify(request)))).toEqual(request)
  })
  it.each([
    { ...request, bundle: "relative/Magnitude.app" },
    { ...request, bundle: "/Applications/../Magnitude.app" },
    { ...request, archive: "/tmp/file\0.zip" },
    { ...request, executable: "/bin/sh" },
    { ...request, offer: { ...request.offer, release: { ...request.offer.release, bytes: -1 } } },
  ])("rejects invalid paths and excess fields", async input => {
    expect(await Effect.runPromise(decodeMacArchiveInstallerRequest(JSON.stringify(input)).pipe(Effect.isFailure))).toBe(true)
  })
  it("rejects oversized input", async () => {
    expect(await Effect.runPromise(decodeMacArchiveInstallerRequest(" ".repeat(16385)).pipe(Effect.isFailure))).toBe(true)
  })
})
