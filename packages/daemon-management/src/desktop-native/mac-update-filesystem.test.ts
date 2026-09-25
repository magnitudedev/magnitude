import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Scope } from "effect"
import { createRequire } from "node:module"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { MacUpdateFilesystem, nativeMacUpdateFilesystem } from "./mac-update-filesystem"

const addon = fileURLToPath(new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url))
const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | CommandExecutor.CommandExecutor | MacUpdateFilesystem | Scope.Scope>) =>
  Effect.runPromise(effect.pipe(Effect.scoped, Effect.provide([nativeMacUpdateFilesystem(addon), BunContext.layer])))
const fixture = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const native = yield* MacUpdateFilesystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-update-filesystem-" })
  const stagePath = join(root, "transaction")
  yield* fs.makeDirectory(stagePath, { mode: 0o700 })
  const parent = yield* native.open(root, false)
  const stage = yield* native.open(stagePath, true)
  return { fs, native, root, stagePath, parent, stage }
})

describe.skipIf(process.platform !== "darwin")("native macOS transaction filesystem", () => {
  it("publishes a fresh bundle once without changing its identity", () => run(Effect.gen(function* () {
    const { fs, native, stagePath, parent, stage } = yield* fixture
    yield* fs.makeDirectory(join(stagePath, "Magnitude.app"))
    const replacement = Option.getOrThrow(yield* native.inspect(stage, "Magnitude.app"))
    expect(yield* native.publish(parent, "Magnitude.app", stage, "Magnitude.app", parent.identity).pipe(Effect.isFailure)).toBe(true)
    yield* native.publish(parent, "Magnitude.app", stage, "Magnitude.app", replacement)
    expect(Option.getOrThrow(yield* native.inspect(parent, "Magnitude.app"))).toBe(replacement)
    expect(Option.isNone(yield* native.inspect(stage, "Magnitude.app"))).toBe(true)
    expect(yield* native.publish(parent, "Magnitude.app", stage, "Magnitude.app", replacement).pipe(Effect.isFailure)).toBe(true)
  })))

  it.each(["directory", "file", "symlink"])("fresh publication preserves an existing %s", kind => run(Effect.gen(function* () {
    const { fs, native, root, stagePath, parent, stage } = yield* fixture
    const destination = join(root, "Magnitude.app")
    if (kind === "directory") yield* fs.makeDirectory(destination)
    else if (kind === "file") yield* fs.writeFileString(destination, "keep")
    else yield* fs.symlink("missing", destination)
    yield* fs.makeDirectory(join(stagePath, "Magnitude.app"))
    const replacement = Option.getOrThrow(yield* native.inspect(stage, "Magnitude.app"))
    expect(yield* native.publish(parent, "Magnitude.app", stage, "Magnitude.app", replacement).pipe(Effect.isFailure)).toBe(true)
    expect(Option.getOrThrow(yield* native.inspect(stage, "Magnitude.app"))).toBe(replacement)
    if (kind === "directory") expect(yield* fs.readDirectory(destination)).toEqual([])
    else if (kind === "file") expect(yield* fs.readFileString(destination)).toBe("keep")
    else expect(yield* fs.readLink(destination)).toBe("missing")
  })))

  it("durably replaces a private record and reads the exact bytes", () => run(Effect.gen(function* () {
    const { fs, native, stage, stagePath } = yield* fixture
    expect(Option.isNone(yield* native.readRecord(stage))).toBe(true)
    yield* native.writeRecord(stage, Buffer.from('{"state":"Intent"}'))
    yield* native.writeRecord(stage, Buffer.from('{"state":"Committed"}'))
    expect(Buffer.from(Option.getOrThrow(yield* native.readRecord(stage))).toString()).toBe('{"state":"Committed"}')
    expect(yield* fs.readDirectory(stagePath)).toEqual(["transaction.json"])
    expect((yield* fs.stat(join(stagePath, "transaction.json"))).mode & 0o777).toBe(0o600)
  })))

  it("exchanges only expected directory identities and refuses replay", () => run(Effect.gen(function* () {
    const { fs, native, root, stagePath, parent, stage } = yield* fixture
    yield* fs.makeDirectory(join(root, "Magnitude.app"))
    yield* fs.makeDirectory(join(stagePath, "Magnitude.app"))
    const previous = Option.getOrThrow(yield* native.inspect(parent, "Magnitude.app"))
    const replacement = Option.getOrThrow(yield* native.inspect(stage, "Magnitude.app"))
    yield* native.exchange(parent, "Magnitude.app", previous, stage, "Magnitude.app", replacement)
    expect(Option.getOrThrow(yield* native.inspect(parent, "Magnitude.app"))).toBe(replacement)
    expect(Option.getOrThrow(yield* native.inspect(stage, "Magnitude.app"))).toBe(previous)
    expect(yield* native.exchange(parent, "Magnitude.app", previous, stage, "Magnitude.app", replacement).pipe(Effect.isFailure)).toBe(true)
    expect(Option.getOrThrow(yield* native.inspect(parent, "Magnitude.app"))).toBe(replacement)
  })))

  it("refuses a substituted parent without writing into either location", () => run(Effect.gen(function* () {
    const { fs, native, stagePath, stage } = yield* fixture
    yield* fs.rename(stagePath, `${stagePath}-retained`)
    yield* fs.makeDirectory(stagePath, { mode: 0o700 })
    expect(yield* native.writeRecord(stage, Buffer.from("new")).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readDirectory(stagePath)).toEqual([])
    expect(yield* fs.readDirectory(`${stagePath}-retained`)).toEqual([])
  })))

  it("distinguishes absence from a symlink or file at a bundle name", () => run(Effect.gen(function* () {
    const { fs, native, root, parent } = yield* fixture
    expect(Option.isNone(yield* native.inspect(parent, "missing"))).toBe(true)
    yield* fs.writeFileString(join(root, "file"), "data")
    yield* fs.symlink("transaction", join(root, "linked"))
    for (const name of ["file", "linked", "..", "transaction/child", "name\0truncated"]) {
      expect(yield* native.inspect(parent, name).pipe(Effect.isFailure)).toBe(true)
    }
  })))

  it("refuses unsafe existing records without replacing them", () => run(Effect.gen(function* () {
    const { fs, native, root, stagePath, stage } = yield* fixture
    const original = join(root, "retain")
    yield* fs.writeFileString(original, "retained")
    yield* fs.symlink(original, join(stagePath, "transaction.json"))
    expect(yield* native.readRecord(stage).pipe(Effect.isFailure)).toBe(true)
    expect(yield* native.writeRecord(stage, Buffer.from("replacement")).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readFileString(original)).toBe("retained")
    expect(yield* fs.readLink(join(stagePath, "transaction.json"))).toBe(original)
  })))

  it("rejects oversized records and widened directory permissions", () => run(Effect.gen(function* () {
    const { fs, native, stagePath, stage } = yield* fixture
    expect(yield* native.writeRecord(stage, Buffer.alloc(16385)).pipe(Effect.isFailure)).toBe(true)
    yield* fs.chmod(stagePath, 0o755)
    expect(yield* native.writeRecord(stage, Buffer.from("new")).pipe(Effect.isFailure)).toBe(true)
    expect(yield* native.open(stagePath, true).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readDirectory(stagePath)).toEqual([])
  })))

  it("refuses an extended access grant despite private mode bits", () => run(Effect.gen(function* () {
    const { native, stagePath, stage } = yield* fixture
    expect(yield* Command.make("/bin/chmod", "+a", "everyone allow read,search", stagePath).pipe(Command.exitCode)).toBe(0)
    expect(yield* native.open(stagePath, true).pipe(Effect.isFailure)).toBe(true)
    expect(yield* native.writeRecord(stage, Buffer.from("new")).pipe(Effect.isFailure)).toBe(true)
  })))

  it("refuses hard-linked or oversized records", () => run(Effect.gen(function* () {
    const { fs, native, root, stagePath, stage } = yield* fixture
    const record = join(stagePath, "transaction.json")
    yield* fs.writeFileString(record, "old", { mode: 0o600 })
    yield* fs.link(record, join(root, "linked-record"))
    expect(yield* native.readRecord(stage).pipe(Effect.isFailure)).toBe(true)
    expect(yield* native.writeRecord(stage, Buffer.from("new")).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readFileString(record)).toBe("old")
    yield* fs.remove(record)
    yield* fs.writeFile(record, Buffer.alloc(16385), { mode: 0o600 })
    expect(yield* native.readRecord(stage).pipe(Effect.isFailure)).toBe(true)
    expect(yield* native.writeRecord(stage, Buffer.from("new")).pipe(Effect.isFailure)).toBe(true)
  })))

  it("requires native capabilities and makes close idempotent", () => run(Effect.gen(function* () {
    const { root } = yield* fixture
    yield* Effect.sync(() => {
      const native = createRequire(import.meta.url)(addon)
      expect(() => native.closeMacUpdateDirectory({})).toThrow()
      const retained = native.openMacUpdateDirectory(root, true)
      expect(() => native.closeMacUpdateDirectory(Object.create(retained))).toThrow()
      expect(() => native.writeMacUpdateRecord(retained, "not bytes")).toThrow()
      native.closeMacUpdateDirectory(retained)
      native.closeMacUpdateDirectory(retained)
      expect(() => native.readMacUpdateRecord(retained)).toThrow()
    })
  })))

  it("synchronizes the expected staged tree without following links", () => run(Effect.gen(function* () {
    const { fs, native, root, stagePath, stage } = yield* fixture
    const bundle = join(stagePath, "Magnitude.app")
    yield* fs.makeDirectory(join(bundle, "Contents"), { recursive: true })
    yield* fs.writeFileString(join(bundle, "Contents/data"), "staged")
    yield* fs.symlink(join(root, "missing-external-target"), join(bundle, "link"))
    const identity = Option.getOrThrow(yield* native.inspect(stage, "Magnitude.app"))
    yield* native.syncTree(stage, "Magnitude.app", identity)
    expect(yield* native.syncTree(stage, "Magnitude.app", stage.identity).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readLink(join(bundle, "link"))).toBe(join(root, "missing-external-target"))
  })))

  it("refuses special files and hard links during staged synchronization", () => run(Effect.gen(function* () {
    const { fs, native, root, stagePath, stage } = yield* fixture
    const bundle = join(stagePath, "Magnitude.app")
    yield* fs.makeDirectory(bundle)
    const identity = Option.getOrThrow(yield* native.inspect(stage, "Magnitude.app"))
    expect(yield* Command.make("/usr/bin/mkfifo", join(bundle, "pipe")).pipe(Command.exitCode)).toBe(0)
    expect(yield* native.syncTree(stage, "Magnitude.app", identity).pipe(Effect.isFailure)).toBe(true)
    yield* fs.remove(join(bundle, "pipe"))
    yield* fs.writeFileString(join(root, "external"), "outside")
    yield* fs.link(join(root, "external"), join(bundle, "linked"))
    expect(yield* native.syncTree(stage, "Magnitude.app", identity).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readFileString(join(root, "external"))).toBe("outside")
  })))
  it("removes only the expected private tree without traversing outside links", () => run(Effect.gen(function* () {
    const { fs, native, root, stagePath, stage, parent } = yield* fixture
    const bundle = join(stagePath, "Magnitude.app")
    const external = join(root, "external")
    yield* fs.makeDirectory(join(bundle, "Contents"), { recursive: true })
    yield* fs.makeDirectory(external)
    yield* fs.writeFileString(join(external, "retain"), "outside")
    yield* fs.symlink(external, join(bundle, "Contents/link"))
    yield* fs.link(join(external, "retain"), join(bundle, "Contents/hardlink"))
    yield* fs.writeFileString(join(bundle, "Contents/data"), "displaced")
    const identity = Option.getOrThrow(yield* native.inspect(stage, "Magnitude.app"))
    expect(yield* native.removeTree(stage, "Magnitude.app", parent.identity).pipe(Effect.isFailure)).toBe(true)
    expect(yield* native.removeTree(parent, "transaction", stage.identity).pipe(Effect.isFailure)).toBe(true)
    yield* native.writeRecord(stage, Buffer.from("retained receipt"))
    yield* native.removeTree(stage, "Magnitude.app", identity)
    expect(yield* fs.readDirectory(stagePath)).toEqual(["transaction.json"])
    expect(yield* fs.readFileString(join(external, "retain"))).toBe("outside")
    expect(Buffer.from(Option.getOrThrow(yield* native.readRecord(stage))).toString()).toBe("retained receipt")
  })))

  it("durably removes only the exact retained receipt", () => run(Effect.gen(function* () {
    const { native, stage } = yield* fixture
    const receipt = Buffer.from("terminal receipt")
    yield* native.writeRecord(stage, receipt)
    expect(yield* native.removeRecord(stage, Buffer.from("different receipt")).pipe(Effect.isFailure)).toBe(true)
    expect(Buffer.from(Option.getOrThrow(yield* native.readRecord(stage)))).toEqual(receipt)
    yield* native.removeRecord(stage, receipt)
    expect(Option.isNone(yield* native.readRecord(stage))).toBe(true)
  })))

})
