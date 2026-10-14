import { generateKeyPairSync } from "node:crypto"
import { Effect, Option, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { UpdateManifest, signUpdateManifest } from "../../../release/src/hosted-update/manifest"
import { PreparedUpdateStore, UpdatePreferences } from "@magnitudedev/daemon-management/desktop-native"
import { WindowsInstallerSignatureFailed } from "../desktop-native/windows-update-signature"
import { PrivateFilePermissionsFailed } from "../desktop-native/private-files"
import { MacBundleVerificationFailed } from "../desktop-native/mac-update-validation"
import { MacUpdateFilesystemFailed } from "../desktop-native/mac-update-filesystem"
import { UpdateIdentityFailed } from "./update-identity"
import { ApplicationUpdateFailed, ApplicationUpdateSource, makeApplicationUpdate } from "./application-update"

const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, tag: "@magnitudedev/cli@2.0.0", version: "2.0.0", commit: "a".repeat(40), artifact: {
  id: "windows", target: { os: "windows", arch: "x64", package: "windows-exe" }, filename: "desktop.exe", bytes: 100, sha256: "a".repeat(64),
} })
const candidate = (await Effect.runPromise(signUpdateManifest(manifest, generateKeyPairSync("ed25519").privateKey))).release

describe("update failure messages", () => {
  it.each([
    new WindowsInstallerSignatureFailed(), new PrivateFilePermissionsFailed(), new MacBundleVerificationFailed(),
    new MacUpdateFilesystemFailed(), new UpdateIdentityFailed(),
  ])("$_tag explains itself in words", error => {
    expect(error.message).toMatch(/^[A-Z].*\.$/)
    expect(String(error)).not.toContain("{")
  })

  it("reports a rejected Windows publisher signature as the transfer failure", () => Effect.runPromise(Effect.gen(function* () {
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: () => Effect.succeed(Option.some(candidate)),
      download: () => Effect.succeed("archive"),
      stage: () => new WindowsInstallerSignatureFailed().pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message }))),
    }))
    yield* owner.check("manual")
    yield* owner.changes.pipe(Stream.filter(state => state.transfer._tag === "Available"), Stream.take(1), Stream.runDrain)
    yield* owner.download
    yield* owner.changes.pipe(Stream.filter(state => state.transfer._tag === "Failed"), Stream.take(1), Stream.runDrain)
    expect((yield* owner.state).transfer).toEqual({ _tag: "Failed", message: "The downloaded update's publisher signature didn't verify, so it wasn't installed." })
  }).pipe(
    Effect.provideService(PreparedUpdateStore, { read: Effect.succeed(Option.none()), prepare: () => Effect.void, verify: () => Effect.die("Unexpected verification"), recordAttempt: () => Effect.void, recordFailure: () => Effect.void, complete: () => Effect.void, discard: Effect.void, removeAbandonedTransfers: Effect.void, outcome: Effect.succeed(Option.none()), recordOutcome: () => Effect.void, markOutcomeReported: Effect.void, hold: Effect.succeed(Option.none()), recordHold: () => Effect.void, releaseHold: Effect.void }),
    Effect.provideService(UpdatePreferences, { read: Effect.succeed(false), write: () => Effect.void }),
    Effect.scoped, Effect.timeout("3 seconds"))))
})
