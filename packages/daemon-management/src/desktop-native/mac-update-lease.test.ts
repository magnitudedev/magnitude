import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Scope, Stream } from "effect"
import { createRequire } from "node:module"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { MacUpdateAdmission, nativeMacUpdateAdmission } from "./mac-update-lease"

const addon = fileURLToPath(new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url))
const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | CommandExecutor.CommandExecutor | MacUpdateAdmission | Scope.Scope>) =>
  Effect.runPromise(effect.pipe(Effect.scoped, Effect.provide([nativeMacUpdateAdmission(addon), BunContext.layer])))
const fixture = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const admission = yield* MacUpdateAdmission
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-update-admission-" })
  const bundle = join(root, "Magnitude.app")
  const lock = join(root, ".Magnitude.app.installation.lock")
  yield* fs.makeDirectory(bundle)
  return { fs, admission, root, bundle, lock }
})

describe.skipIf(process.platform !== "darwin")("macOS installation admission", () => {
  it("admits multiple readers and excludes replacement until every reader retires", () => run(Effect.gen(function* () {
    const { admission, bundle, fs, lock } = yield* fixture
    yield* Effect.scoped(Effect.gen(function* () {
      expect(Option.isSome(yield* admission.shared(bundle))).toBe(true)
      expect(Option.isSome(yield* admission.shared(bundle))).toBe(true)
      expect(Option.isNone(yield* admission.exclusive(bundle))).toBe(true)
    }))
    const identity = (yield* fs.stat(lock)).ino
    const exclusive = Option.getOrThrow(yield* admission.exclusive(bundle))
    yield* exclusive.validate
    expect(Option.isNone(yield* admission.shared(bundle))).toBe(true)
    expect(Option.isNone(yield* admission.exclusive(bundle))).toBe(true)
    expect((yield* fs.stat(lock)).ino).toEqual(identity)
  })))

  it("excludes a separate process and releases admission after its death", () => run(Effect.gen(function* () {
    const { admission, bundle, fs, lock } = yield* fixture
    const child = yield* Command.make(process.execPath,
      fileURLToPath(new URL("./fixtures/mac-update-lease.cjs", import.meta.url)), addon, bundle, "Shared").pipe(Command.start)
    const ready = yield* child.stdout.pipe(Stream.decodeText(), Stream.splitLines, Stream.take(1), Stream.runHead,
      Effect.timeout("10 seconds"))
    expect(Option.getOrThrow(ready)).toBe("ready")
    const identity = (yield* fs.stat(lock)).ino
    expect(Option.isNone(yield* admission.exclusive(bundle))).toBe(true)
    yield* child.kill("SIGKILL")
    expect(String(yield* child.exitCode.pipe(Effect.flip))).toContain("SIGKILL")
    expect(Option.isSome(yield* admission.exclusive(bundle))).toBe(true)
    expect((yield* fs.stat(lock)).ino).toEqual(identity)
  })))

  it("keeps the same lock after replacing the bundle", () => run(Effect.gen(function* () {
    const { fs, admission, root, bundle, lock } = yield* fixture
    const retained = Option.getOrThrow(yield* admission.exclusive(bundle))
    const identity = (yield* fs.stat(lock)).ino
    yield* fs.rename(bundle, join(root, "previous.app"))
    yield* fs.makeDirectory(bundle)
    yield* retained.validate
    expect(Option.isNone(yield* admission.shared(bundle))).toBe(true)
    expect((yield* fs.stat(lock)).ino).toEqual(identity)
  })))

  it("detects replacement of a held lock or parent", () => run(Effect.gen(function* () {
    const { fs, admission, bundle, lock, root } = yield* fixture
    const retained = Option.getOrThrow(yield* admission.exclusive(bundle))
    yield* fs.rename(lock, `${lock}.retained`)
    yield* fs.writeFileString(lock, "", { mode: 0o644 })
    expect(yield* retained.validate.pipe(Effect.isFailure)).toBe(true)
    yield* fs.remove(lock)
    yield* fs.rename(`${lock}.retained`, lock)
    yield* retained.validate
    // The scoped root remains in place for cleanup; only its child parent is substituted.
    const child = join(root, "child")
    yield* fs.makeDirectory(join(child, "Magnitude.app"), { recursive: true })
    const nested = Option.getOrThrow(yield* admission.shared(join(child, "Magnitude.app")))
    yield* fs.rename(child, `${child}.retained`)
    yield* fs.makeDirectory(child)
    expect(yield* nested.validate.pipe(Effect.isFailure)).toBe(true)
  })))

  it("refuses unsafe lock files without rewriting them", () => run(Effect.gen(function* () {
    const { fs, admission, bundle, lock, root } = yield* fixture
    yield* fs.writeFileString(lock, "retained", { mode: 0o644 })
    expect(yield* admission.shared(bundle).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readFileString(lock)).toBe("retained")
    yield* fs.writeFileString(lock, "")
    yield* fs.chmod(lock, 0o666)
    expect(yield* admission.shared(bundle).pipe(Effect.isFailure)).toBe(true)
    yield* fs.chmod(lock, 0o644)
    yield* fs.link(lock, join(root, "linked"))
    expect(yield* admission.shared(bundle).pipe(Effect.isFailure)).toBe(true)
    yield* fs.remove(lock)
    yield* fs.symlink("linked", lock)
    expect(yield* admission.shared(bundle).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readLink(lock)).toBe("linked")
  })))

  it("rejects extended grants even when mode bits remain read-only for other users", () => run(Effect.gen(function* () {
    const { fs, admission, bundle, lock } = yield* fixture
    yield* Effect.scoped(admission.shared(bundle))
    expect(yield* Command.make("/bin/chmod", "+a", "everyone allow write", lock).pipe(Command.exitCode)).toBe(0)
    expect((yield* fs.stat(lock)).mode & 0o777).toBe(0o644)
    expect(yield* admission.exclusive(bundle).pipe(Effect.isFailure)).toBe(true)
  })))

  it("rejects unsafe bundle paths and requires exact native release capabilities", () => run(Effect.gen(function* () {
    const { fs, admission, bundle, root } = yield* fixture
    yield* fs.symlink(bundle, join(root, "alias.app"))
    for (const path of [join(root, "alias.app"), `${bundle}\0suffix`, root + "/..", bundle + "/"]) {
      expect(yield* admission.shared(path).pipe(Effect.isFailure)).toBe(true)
    }
    yield* Effect.sync(() => {
      const native = createRequire(import.meta.url)(addon)
      const retained = native.acquireMacUpdateLease(bundle, false)
      expect(() => native.releaseMacUpdateLease({})).toThrow()
      expect(() => native.releaseMacUpdateLease(Object.create(retained))).toThrow()
      native.releaseMacUpdateLease(retained)
      native.releaseMacUpdateLease(retained)
      expect(() => native.validateMacUpdateLease(retained)).toThrow()
    })
  })))
})
