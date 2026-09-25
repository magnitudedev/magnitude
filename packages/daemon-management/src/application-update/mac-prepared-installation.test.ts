import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Ref, Scope } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../../release/src/hosted-update/release"
import { PreparedUpdateStore, PreparedUpdateFailed, type PreparedUpdate } from "../desktop-native/prepared-update"
import { nativeMacUpdateFilesystem, MacUpdateFilesystem } from "../desktop-native/mac-update-filesystem"
import { nativeMacUpdateAdmission, MacUpdateAdmission } from "../desktop-native/mac-update-lease"
import { MacApplicationInstallation } from "../desktop-native/mac-update-installation"
import { MacBundleVerifier, MacBundleVerificationFailed } from "../desktop-native/mac-update-validation"
import { MacUpdateArchiveStager, MacUpdateStagingFailed } from "../desktop-native/mac-update-staging"
import { completeMacPreparedInstallation, recoverMacPreparedInstallation } from "./mac-prepared-installation"
import { installMacApplicationArchive } from "./mac-archive-installation"
import { GuardedCommand, GuardedCommandFailed } from "../desktop-native/guarded-command"
const addon = fileURLToPath(new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url))
const keys = generateKeyPairSync("ed25519")
const release = await Effect.runPromise(signUpdateRelease({ version: "0.1.6", bytes: 1, sha256: "0".repeat(64) },
  { os: "darwin", arch: "arm64", package: "mac-zip" }, keys.privateKey))
const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | MacUpdateFilesystem | MacUpdateAdmission | Scope.Scope>) =>
  Effect.runPromise(effect.pipe(Effect.scoped, Effect.provide([BunContext.layer, nativeMacUpdateFilesystem(addon), nativeMacUpdateAdmission(addon)])))
const fixture = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-mac-install-" })
  const bundle = join(root, "Magnitude.app")
  yield* fs.makeDirectory(bundle)
  yield* fs.writeFileString(join(bundle, "version"), "0.1.5")
  const events: string[] = []
  const step = (name: string) => Effect.sync(() => { events.push(name) })
  const prepared = yield* Ref.make<Option.Option<PreparedUpdate>>(Option.some({ release, installation: { _tag: "Unattempted" } }))
  const store = PreparedUpdateStore.of({
    read: Ref.get(prepared),
    verify: () => step("verify").pipe(Effect.as("archive")),
    recordAttempt: () => step("attempt").pipe(Effect.zipRight(Ref.set(prepared, Option.some({ release, installation: { _tag: "Attempted" } })))),
    recordFailure: (_, reason) => step("failure").pipe(Effect.zipRight(Ref.set(prepared, Option.some({ release, installation: { _tag: "Failed", reason } })))),
    discard: step("discard").pipe(Effect.zipRight(Ref.set(prepared, Option.none()))), removeAbandonedTransfers: Effect.void, outcome: Effect.succeed(Option.none()), recordOutcome: () => Effect.void, markOutcomeReported: Effect.void,
    prepare: () => Effect.die("Unexpected download"),
  })
  const stager = MacUpdateArchiveStager.of({ stage: (_, staging) => Effect.gen(function* () {
    yield* step("stage")
    yield* fs.makeDirectory(join(staging.path, "Magnitude.app"))
    yield* fs.writeFileString(join(staging.path, "Magnitude.app/version"), "0.1.6")
  }).pipe(Effect.mapError(() => new MacUpdateStagingFailed())) })
  const verifier = MacBundleVerifier.of({ verify: (path, expected) => fs.readFileString(join(path, "version")).pipe(
    Effect.filterOrFail(version => version === expected.version, () => new MacBundleVerificationFailed()),
    Effect.mapError(() => new MacBundleVerificationFailed()), Effect.asVoid) })
  const install = completeMacPreparedInstallation({ bundle, version: "0.1.5", architecture: "arm64" }).pipe(
    Effect.provideService(MacBundleVerifier, verifier), Effect.provideService(MacApplicationInstallation, { isInstalling: () => Effect.succeed(false) }))
  const recover = recoverMacPreparedInstallation(bundle).pipe(
    Effect.provideService(MacBundleVerifier, verifier), Effect.provideService(MacApplicationInstallation, { isInstalling: () => Effect.succeed(false) }))
  const archiveInstall = installMacApplicationArchive({ bundle, archive: "archive", release, architecture: "arm64" }).pipe(
    Effect.provideService(MacBundleVerifier, verifier), Effect.provideService(MacUpdateArchiveStager, stager),
    Effect.provideService(MacApplicationInstallation, { isInstalling: () => Effect.succeed(false) }),
    Effect.provideService(GuardedCommand, { run: () => fs.readFileString(join(bundle, "version")).pipe(
      Effect.map(stdout => ({ code: 0, stdout, stderr: "" })),
      Effect.mapError(() => new GuardedCommandFailed({ message: "Missing version" }))) }))
  return { fs, bundle, root, events, store, stager, install, recover, archiveInstall }
})
describe.skipIf(process.platform !== "darwin")("prepared macOS installation", () => {
  it("installs a fresh archive, repeats installation and leaves no transaction workspace", () => run(Effect.gen(function* () {
    const f = yield* fixture
    yield* f.fs.remove(f.bundle, { recursive: true })
    for (let attempt = 0; attempt < 2; attempt++) {
      expect(yield* f.archiveInstall).toEqual({ _tag: "Installed", version: "0.1.6" })
      expect(yield* f.fs.readFileString(join(f.bundle, "version"))).toBe("0.1.6")
      expect(yield* f.fs.exists(join(f.root, ".Magnitude.app.update"))).toBe(false)
    }
    expect(f.events).toEqual(["stage", "stage"])
  })))

  it("archive installation refuses an active installation lease without staging", () => run(Effect.gen(function* () {
    const f = yield* fixture
    const admission = yield* MacUpdateAdmission
    expect(Option.isSome(yield* admission.shared(f.bundle))).toBe(true)
    expect(yield* f.archiveInstall.pipe(Effect.isFailure)).toBe(true)
    expect(f.events).toEqual([])
    expect(yield* f.fs.readFileString(join(f.bundle, "version"))).toBe("0.1.5")
  })))

  it("verifies and records the attempt before staging, then retires preparation after verified exchange", () => run(Effect.gen(function* () {
    const f = yield* fixture
    expect(yield* f.install.pipe(Effect.provideService(PreparedUpdateStore, f.store), Effect.provideService(MacUpdateArchiveStager, f.stager)))
      .toEqual({ _tag: "Installed", version: "0.1.6" })
    expect(f.events).toEqual(["verify", "attempt", "stage", "discard"])
    expect(yield* f.fs.readFileString(join(f.bundle, "version"))).toBe("0.1.6")
    expect(yield* f.fs.exists(join(f.root, ".Magnitude.app.update"))).toBe(false)
  })))
  it("retains a committed transaction if preparation retirement fails and reconciles without staging again", () => run(Effect.gen(function* () {
    const f = yield* fixture
    const failing = { ...f.store, discard: Effect.fail(new PreparedUpdateFailed({ message: "Retirement failed" })) }
    expect(yield* f.install.pipe(Effect.provideService(PreparedUpdateStore, failing), Effect.provideService(MacUpdateArchiveStager, f.stager), Effect.isFailure)).toBe(true)
    expect(yield* f.fs.readFileString(join(f.bundle, "version"))).toBe("0.1.6")
    expect(yield* f.recover.pipe(Effect.provideService(PreparedUpdateStore, f.store)))
      .toEqual({ _tag: "Installed", version: "0.1.6" })
    expect(f.events.filter(event => event === "stage")).toHaveLength(1)
    expect(f.events).toEqual(["verify", "attempt", "stage", "failure", "discard"])
    expect(yield* f.store.read).toEqual(Option.none())
  })))
  it("retains the old installation after partial extraction and cleans staging on explicit retry", () => run(Effect.gen(function* () {
    const f = yield* fixture
    const failedStage = MacUpdateArchiveStager.of({ stage: (...args) => f.stager.stage(...args).pipe(
      Effect.zipRight(new MacUpdateStagingFailed())) })
    expect(yield* f.install.pipe(Effect.provideService(PreparedUpdateStore, f.store),
      Effect.provideService(MacUpdateArchiveStager, failedStage), Effect.isFailure)).toBe(true)
    expect(yield* f.fs.readFileString(join(f.bundle, "version"))).toBe("0.1.5")
    expect(f.events).toEqual(["verify", "attempt", "stage", "failure"])
    const retained = yield* f.store.read
    expect(Option.getOrThrow(retained).installation._tag).toBe("Failed")
    expect(yield* f.recover.pipe(Effect.provideService(PreparedUpdateStore, f.store))).toEqual({ _tag: "NoTransaction" })
    expect(yield* f.store.read).toEqual(retained)
    expect(f.events).toEqual(["verify", "attempt", "stage", "failure"])
    expect(yield* f.fs.readFileString(join(f.bundle, "version"))).toBe("0.1.5")
    expect(yield* f.install.pipe(Effect.provideService(PreparedUpdateStore, f.store),
      Effect.provideService(MacUpdateArchiveStager, f.stager))).toEqual({ _tag: "Installed", version: "0.1.6" })
    expect(yield* f.fs.exists(join(f.root, ".Magnitude.app.update"))).toBe(false)
  })))
  it("does not create a workspace or install unattempted preparation during recovery", () => run(Effect.gen(function* () {
    const f = yield* fixture
    expect(yield* f.recover.pipe(Effect.provideService(PreparedUpdateStore, f.store))).toEqual({ _tag: "NoTransaction" })
    expect(f.events).toEqual([])
    expect(yield* f.fs.exists(join(f.root, ".Magnitude.app.update"))).toBe(false)
    expect(yield* f.fs.readFileString(join(f.bundle, "version"))).toBe("0.1.5")
  })))
  it("does not record an attempt or stage after archive verification fails", () => run(Effect.gen(function* () {
    const f = yield* fixture
    expect(yield* f.install.pipe(Effect.provideService(PreparedUpdateStore, { ...f.store,
      verify: () => Effect.fail(new PreparedUpdateFailed({ message: "Invalid archive" })) }),
      Effect.provideService(MacUpdateArchiveStager, f.stager), Effect.isFailure)).toBe(true)
    expect(f.events).toEqual([])
    expect(yield* f.fs.readFileString(join(f.bundle, "version"))).toBe("0.1.5")
  })))
})
