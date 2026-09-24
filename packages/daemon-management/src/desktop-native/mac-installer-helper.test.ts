import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Scope } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { basename, dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { MacUpdateAdmission, nativeMacUpdateAdmission } from "./mac-update-lease"
import { MacUpdateFilesystem, nativeMacUpdateFilesystem } from "./mac-update-filesystem"
import { MacBundleVerifier, MacBundleVerificationFailed } from "./mac-update-validation"
import { MacInstallerCodeVerifier, MacInstallerHelperFailed, prepareMacInstallerHelper } from "./mac-installer-helper"
import { readInstalledUpdateConfiguration } from "../application-update/update-configuration"
const addon = fileURLToPath(new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url))
const publicKey = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" }).toString()
const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | MacUpdateFilesystem | MacUpdateAdmission | Scope.Scope>) =>
  Effect.runPromise(effect.pipe(Effect.scoped, Effect.provide([BunContext.layer, nativeMacUpdateFilesystem(addon), nativeMacUpdateAdmission(addon)])))
const fixture = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const admission = yield* MacUpdateAdmission
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-installer-helper-" })
  const bundle = join(root, "Magnitude.app"), resources = join(bundle, "Contents/Resources"), stateDirectory = join(root, "state")
  yield* fs.makeDirectory(resources, { recursive: true })
  yield* fs.makeDirectory(stateDirectory, { mode: 0o700 })
  for (const name of ["magnitude", "desktop-host.node", "magnitude-command", "magnitude-extract"]) yield* fs.writeFileString(join(resources, name), name)
  yield* fs.writeFileString(join(resources, "update-configuration.json"), JSON.stringify({ origin: "https://magnitude.dev", keyId: "fixture", publicKey, acceptance: false }))
  const lease = Option.getOrThrow(yield* admission.exclusive(bundle))
  const checked: string[] = []
  const bundleVerifier = MacBundleVerifier.of({ verify: (path, expected) => Effect.sync(() => {
    expect(path).toBe(bundle); expect(expected.version).toBe("0.1.5"); checked.push("bundle")
  }) })
  const codeVerifier = MacInstallerCodeVerifier.of({ verify: path => Effect.gen(function* () {
    expect(yield* fs.readFileString(path)).toBe(basename(path)); checked.push(basename(path))
  }).pipe(Effect.mapError(() => new MacInstallerHelperFailed())) })
  const prepare = prepareMacInstallerHelper({ resources, stateDirectory, lease, version: "0.1.5", architecture: "arm64" })
  return { fs, root, resources, stateDirectory, lease, checked, bundleVerifier, codeVerifier, prepare }
})
describe.skipIf(process.platform !== "darwin")("private macOS installer runtime", () => {
  it("verifies copied runtime files and retains installed trust outside the replaced bundle", () => run(Effect.gen(function* () {
    const f = yield* fixture
    let directory = ""
    yield* Effect.scoped(Effect.gen(function* () {
      const helper = yield* f.prepare.pipe(Effect.provideService(MacBundleVerifier, f.bundleVerifier), Effect.provideService(MacInstallerCodeVerifier, f.codeVerifier))
      directory = helper.directory
      expect(dirname(directory)).toBe(join(f.stateDirectory, "mac-installers"))
      expect((yield* f.fs.stat(directory)).mode & 0o777).toBe(0o700)
      expect((yield* f.fs.stat(helper.executable)).mode & 0o777).toBe(0o700)
      expect((yield* f.fs.stat(helper.addonPath)).mode & 0o777).toBe(0o600)
      const configuration = yield* readInstalledUpdateConfiguration(directory)
      expect(configuration.trustedPublishers.get("fixture")?.export({ type: "spki", format: "pem" }).toString()).toBe(publicKey)
      expect(f.checked).toEqual(["bundle", "magnitude", "desktop-host.node", "magnitude-command", "magnitude-extract"])
    }))
    expect(yield* f.fs.exists(directory)).toBe(false)
  })))
  it("creates no helper when installed bundle verification fails", () => run(Effect.gen(function* () {
    const f = yield* fixture
    expect(yield* f.prepare.pipe(Effect.provideService(MacBundleVerifier, { verify: () => new MacBundleVerificationFailed() }),
      Effect.provideService(MacInstallerCodeVerifier, f.codeVerifier), Effect.isFailure)).toBe(true)
    expect(yield* f.fs.exists(join(f.stateDirectory, "mac-installers"))).toBe(false)
  })))
  it("retires incomplete helper files after copied code verification fails", () => run(Effect.gen(function* () {
    const f = yield* fixture
    expect(yield* Effect.scoped(f.prepare.pipe(Effect.provideService(MacBundleVerifier, f.bundleVerifier),
      Effect.provideService(MacInstallerCodeVerifier, { verify: () => new MacInstallerHelperFailed() }))).pipe(Effect.isFailure)).toBe(true)
    expect(yield* f.fs.readDirectory(join(f.stateDirectory, "mac-installers"))).toEqual([])
    yield* f.lease.validate
  })))
  it("rejects linked runtime files and unsafe existing helper storage", () => run(Effect.gen(function* () {
    const f = yield* fixture
    const parent = join(f.stateDirectory, "mac-installers")
    yield* f.fs.makeDirectory(parent, { mode: 0o755 })
    expect(yield* Effect.scoped(f.prepare.pipe(Effect.provideService(MacBundleVerifier, f.bundleVerifier),
      Effect.provideService(MacInstallerCodeVerifier, f.codeVerifier))).pipe(Effect.isFailure)).toBe(true)
    expect((yield* f.fs.stat(parent)).mode & 0o777).toBe(0o755)
    yield* f.fs.chmod(parent, 0o700)
    yield* f.fs.remove(join(f.resources, "magnitude"))
    yield* f.fs.symlink(join(f.resources, "magnitude-command"), join(f.resources, "magnitude"))
    expect(yield* Effect.scoped(f.prepare.pipe(Effect.provideService(MacBundleVerifier, f.bundleVerifier),
      Effect.provideService(MacInstallerCodeVerifier, f.codeVerifier))).pipe(Effect.isFailure)).toBe(true)
    expect(yield* f.fs.readDirectory(parent)).toEqual([])
  })))
})
