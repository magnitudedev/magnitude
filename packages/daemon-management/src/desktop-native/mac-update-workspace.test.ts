import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Scope } from "effect"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { MacUpdateAdmission, nativeMacUpdateAdmission } from "./mac-update-lease"
import { MacUpdateFilesystem, nativeMacUpdateFilesystem } from "./mac-update-filesystem"
import { MacBundleVerifier, MacBundleVerificationFailed } from "./mac-update-validation"
import { exchangeMacUpdate } from "./mac-update-transaction"
import { openMacUpdateWorkspace } from "./mac-update-workspace"
const addon = fileURLToPath(new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url))
const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | MacUpdateFilesystem | MacUpdateAdmission | Scope.Scope>) =>
  Effect.runPromise(effect.pipe(Effect.scoped, Effect.provide([BunContext.layer, nativeMacUpdateFilesystem(addon), nativeMacUpdateAdmission(addon)])))
const fixture = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const native = yield* MacUpdateFilesystem
  const admission = yield* MacUpdateAdmission
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-update-workspace-" })
  const bundle = join(root, "Magnitude.app"), stagingPath = join(root, ".Magnitude.app.update")
  yield* fs.makeDirectory(bundle)
  yield* fs.writeFileString(join(bundle, "version"), "0.1.5")
  const verifier = MacBundleVerifier.of({ verify: (path, expected) => fs.readFileString(join(path, "version")).pipe(
    Effect.filterOrFail(version => version === expected.version, () => new MacBundleVerificationFailed()),
    Effect.mapError(() => new MacBundleVerificationFailed()), Effect.asVoid) })
  return { fs, native, admission, root, bundle, stagingPath, verifier }
})
describe.skipIf(process.platform !== "darwin")("macOS installation transaction discovery", () => {
  it("observes without creating staging and excludes owners while retained", () => run(Effect.gen(function* () {
    const { fs, bundle, stagingPath, admission } = yield* fixture
    expect(Option.isNone(yield* Effect.scoped(openMacUpdateWorkspace(bundle, false)))).toBe(true)
    expect(yield* fs.exists(stagingPath)).toBe(false)
    yield* Effect.scoped(Effect.gen(function* () {
      const workspace = Option.getOrThrow(yield* openMacUpdateWorkspace(bundle, true))
      expect(Option.isNone(yield* admission.shared(bundle))).toBe(true)
      yield* workspace.validate
      expect((yield* fs.stat(stagingPath)).mode & 0o777).toBe(0o700)
    }))
    expect(Option.isSome(yield* admission.shared(bundle))).toBe(true)
  })))
  it("uses an already retained installer lease only for its bound installation", () => run(Effect.gen(function* () {
    const { fs, admission, bundle, root } = yield* fixture
    const lease = Option.getOrThrow(yield* admission.exclusive(bundle))
    expect(Option.isSome(yield* openMacUpdateWorkspace(bundle, true, lease))).toBe(true)
    const other = join(root, "Other.app")
    yield* fs.makeDirectory(other)
    expect(yield* openMacUpdateWorkspace(other, true, lease).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.exists(join(root, ".Other.app.update"))).toBe(false)
  })))
  it("defers without staging when an owner retains shared admission", () => run(Effect.gen(function* () {
    const { fs, bundle, stagingPath, admission } = yield* fixture
    yield* admission.shared(bundle)
    expect((yield* openMacUpdateWorkspace(bundle, true).pipe(Effect.flip))._tag).toBe("MacUpdateInstallationBusy")
    expect(yield* fs.exists(stagingPath)).toBe(false)
  })))
  it("discovers a committed exchange in a fresh scope and retires only the displaced bundle", () => run(Effect.gen(function* () {
    const { fs, native, bundle, stagingPath, verifier } = yield* fixture
    yield* Effect.scoped(Effect.gen(function* () {
      const workspace = Option.getOrThrow(yield* openMacUpdateWorkspace(bundle, true))
      const staged = join(stagingPath, "Magnitude.app")
      yield* fs.makeDirectory(staged)
      yield* fs.writeFileString(join(staged, "version"), "0.1.6")
      expect(yield* exchangeMacUpdate(workspace.installed, workspace.installedName, workspace.staging,
        { previous: "0.1.5", replacement: "0.1.6", architecture: "arm64" }).pipe(Effect.provideService(MacBundleVerifier, verifier)))
        .toEqual({ _tag: "Installed", version: "0.1.6" })
      expect(yield* workspace.clearUnpublishedStaging.pipe(Effect.isFailure)).toBe(true)
    }))
    yield* Effect.scoped(Effect.gen(function* () {
      const workspace = Option.getOrThrow(yield* openMacUpdateWorkspace(bundle, false))
      expect(yield* workspace.recover.pipe(Effect.provideService(MacBundleVerifier, verifier))).toEqual({ _tag: "Installed", version: "0.1.6" })
      yield* workspace.retire.pipe(Effect.provideService(MacBundleVerifier, verifier))
      expect(Option.isNone(yield* native.readRecord(workspace.staging))).toBe(true)
      expect(yield* fs.exists(join(stagingPath, "Magnitude.app"))).toBe(false)
    }))
    expect(yield* fs.readFileString(join(bundle, "version"))).toBe("0.1.6")
  })))
  it("clears interrupted extraction without changing the live bundle or following links", () => run(Effect.gen(function* () {
    const { fs, bundle, stagingPath, root } = yield* fixture
    const external = join(root, "retained")
    yield* fs.writeFileString(external, "untouched")
    yield* Effect.scoped(Effect.gen(function* () {
      yield* openMacUpdateWorkspace(bundle, true)
      yield* fs.makeDirectory(join(stagingPath, "Magnitude.app"))
      yield* fs.symlink(external, join(stagingPath, "Magnitude.app/link"))
    }))
    const workspace = Option.getOrThrow(yield* openMacUpdateWorkspace(bundle, false))
    yield* workspace.clearUnpublishedStaging
    expect(yield* fs.exists(join(stagingPath, "Magnitude.app"))).toBe(false)
    expect(yield* fs.readFileString(external)).toBe("untouched")
    expect(yield* fs.readFileString(join(bundle, "version"))).toBe("0.1.5")
  })))
  it("preserves substituted staging and refuses an unrelated directory capability", () => run(Effect.gen(function* () {
    const { fs, native, bundle, stagingPath, root } = yield* fixture
    yield* Effect.scoped(Effect.gen(function* () {
      const workspace = Option.getOrThrow(yield* openMacUpdateWorkspace(bundle, true))
      const unrelatedPath = join(root, "unrelated")
      yield* fs.makeDirectory(unrelatedPath, { mode: 0o700 })
      const unrelated = yield* native.open(unrelatedPath, true)
      expect(yield* native.removeEmptyDirectory(workspace.installed, ".Magnitude.app.update", unrelated).pipe(Effect.isFailure)).toBe(true)
      yield* fs.rename(stagingPath, stagingPath + ".retained")
      yield* fs.makeDirectory(stagingPath, { mode: 0o700 })
      expect(yield* workspace.validate.pipe(Effect.isFailure)).toBe(true)
    }))
    expect(yield* fs.exists(stagingPath)).toBe(true)
    expect(yield* fs.exists(stagingPath + ".retained")).toBe(true)
    expect(yield* fs.exists(join(root, "unrelated"))).toBe(true)
  })))
  it.each(["symlink", "public", "file"])("preserves an unsafe %s discovery entry", kind => run(Effect.gen(function* () {
    const { fs, bundle, stagingPath, root } = yield* fixture
    if (kind === "symlink") yield* fs.symlink(bundle, stagingPath)
    else if (kind === "file") yield* fs.writeFileString(stagingPath, "retained")
    else yield* fs.makeDirectory(stagingPath, { mode: 0o755 })
    expect(yield* openMacUpdateWorkspace(bundle, true).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readFileString(join(bundle, "version"))).toBe("0.1.5")
    expect((yield* fs.readDirectory(root)).includes(".Magnitude.app.update")).toBe(true)
  })))
})
