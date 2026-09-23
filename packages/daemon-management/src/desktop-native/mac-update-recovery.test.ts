import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema, Scope } from "effect"
import { spawn } from "node:child_process"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { MacUpdateFilesystem, MacUpdateFilesystemFailed, nativeMacUpdateFilesystem } from "./mac-update-filesystem"
import { MacBundleVerificationFailed, MacBundleVerifier } from "./mac-update-validation"
import { MacUpdateJournal, recoverMacUpdateTransaction } from "./mac-update-recovery"

const addon = fileURLToPath(new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url))
const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | MacUpdateFilesystem | Scope.Scope>) =>
  Effect.runPromise(effect.pipe(Effect.scoped, Effect.provide([nativeMacUpdateFilesystem(addon), BunContext.layer])))
const fixture = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const native = yield* MacUpdateFilesystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-update-recovery-" })
  const stagingPath = join(root, "transaction")
  yield* fs.makeDirectory(stagingPath, { mode: 0o700 })
  const installed = yield* native.open(root, false)
  const staging = yield* native.open(stagingPath, true)
  const oldPath = join(root, "Magnitude.app"), newPath = join(stagingPath, "Magnitude.app")
  yield* fs.makeDirectory(oldPath)
  yield* fs.makeDirectory(newPath)
  yield* fs.writeFileString(join(oldPath, "version"), "0.1.5")
  yield* fs.writeFileString(join(newPath, "version"), "0.1.6")
  const previous = Option.getOrThrow(yield* native.inspect(installed, "Magnitude.app"))
  const replacement = Option.getOrThrow(yield* native.inspect(staging, "Magnitude.app"))
  const transaction = { id: "867124a0-0656-4c9e-8466-d34447d29bc8", installedParent: installed.identity,
    stagingParent: staging.identity, installedName: "Magnitude.app", architecture: "arm64",
    previous: { identity: previous, version: "0.1.5" }, replacement: { identity: replacement, version: "0.1.6" } }
  // Signature verification has its own native acceptance. Here failures select recovery branches.
  const verifier = MacBundleVerifier.of({ verify: (path, expected) => fs.readFileString(join(path, "version")).pipe(
    Effect.filterOrFail(version => version === expected.version, () => new MacBundleVerificationFailed()),
    Effect.mapError(() => new MacBundleVerificationFailed()), Effect.asVoid) })
  const write = (tag: MacUpdateJournal["_tag"]) => Schema.decodeUnknown(MacUpdateJournal)({ _tag: tag, protocol: 1, transaction }).pipe(
    Effect.flatMap(Schema.encode(Schema.parseJson(MacUpdateJournal))), Effect.flatMap(text => native.writeRecord(staging, Buffer.from(text))))
  const read = native.readRecord(staging).pipe(Effect.flatMap(bytes =>
    Schema.decodeUnknown(Schema.parseJson(MacUpdateJournal))(Buffer.from(Option.getOrThrow(bytes)).toString())))
  const recover = recoverMacUpdateTransaction(installed, "Magnitude.app", staging).pipe(Effect.provideService(MacBundleVerifier, verifier))
  const swap = native.exchange(installed, "Magnitude.app", previous, staging, "Magnitude.app", replacement)
  return { fs, native, root, installed, staging, oldPath, newPath, previous, replacement, write, read, recover, swap }
})

describe.skipIf(process.platform !== "darwin")("macOS transaction recovery on native filesystems", () => {
  it.each(["BeforeExchange", "AfterExchange", "AfterCommit", "BeforeRestore", "AfterRestore"])("recovers after actual process loss at %s", phase => run(Effect.gen(function* () {
    const { root, staging, write, recover } = yield* fixture
    yield* write("ExchangeIntent")
    yield* Effect.async<void>(resume => {
      const child = spawn(process.execPath, [fileURLToPath(new URL("./fixtures/mac-update-recovery-crash.ts", import.meta.url)), root, staging.path, phase], { stdio: ["ignore", "ignore", "inherit"] })
      child.once("error", error => resume(Effect.die(error)))
      child.once("exit", (_code, signal) => resume(Effect.sync(() => expect(signal).toBe("SIGKILL"))))
      return Effect.sync(() => { child.kill("SIGKILL") })
    })
    const expected = phase === "BeforeExchange" ? { _tag: "Preserved", reason: "Interrupted" }
      : phase.endsWith("Restore") ? { _tag: "Preserved", reason: "Restored" } : { _tag: "Installed", version: "0.1.6" }
    expect(yield* recover).toMatchObject(expected)
    expect(yield* recover).toMatchObject(expected)
  })))

  it("leaves an installation with no journal untouched", () => run(Effect.gen(function* () {
    const { recover } = yield* fixture
    expect(yield* recover).toEqual({ _tag: "NoTransaction" })
  })))

  it("abandons an interrupted pre-exchange attempt and never retries it", () => run(Effect.gen(function* () {
    const { write, read, recover, native, installed, previous } = yield* fixture
    yield* write("ExchangeIntent")
    for (let index = 0; index < 2; index++) expect(yield* recover).toEqual({ _tag: "Preserved", version: "0.1.5", reason: "Interrupted" })
    expect((yield* read)._tag).toBe("Abandoned")
    expect(Option.getOrThrow(yield* native.inspect(installed, "Magnitude.app"))).toBe(previous)
  })))

  it("commits an observed exchange without toggling the versions", () => run(Effect.gen(function* () {
    const { write, read, recover, swap, native, installed, staging, previous, replacement } = yield* fixture
    yield* write("ExchangeIntent")
    yield* swap
    for (let index = 0; index < 2; index++) expect(yield* recover).toEqual({ _tag: "Installed", version: "0.1.6" })
    expect((yield* read)._tag).toBe("Committed")
    expect(Option.getOrThrow(yield* native.inspect(installed, "Magnitude.app"))).toBe(replacement)
    expect(Option.getOrThrow(yield* native.inspect(staging, "Magnitude.app"))).toBe(previous)
  })))

  it("restores a verified old bundle when the uncommitted replacement is invalid", () => run(Effect.gen(function* () {
    const { fs, oldPath, write, read, recover, swap, native, installed, previous } = yield* fixture
    yield* write("ExchangeIntent")
    yield* swap
    yield* fs.writeFileString(join(oldPath, "version"), "damaged")
    for (let index = 0; index < 2; index++) expect(yield* recover).toEqual({ _tag: "Preserved", version: "0.1.5", reason: "Restored" })
    expect((yield* read)._tag).toBe("Restored")
    expect(Option.getOrThrow(yield* native.inspect(installed, "Magnitude.app"))).toBe(previous)
  })))

  it.each([false, true])("resumes restore intent with restoration already applied = %s", restored => run(Effect.gen(function* () {
    const { write, read, recover, swap, native, installed, staging, previous, replacement } = yield* fixture
    yield* write("RestoreIntent")
    yield* swap
    if (restored) yield* native.exchange(installed, "Magnitude.app", replacement, staging, "Magnitude.app", previous)
    expect(yield* recover).toEqual({ _tag: "Preserved", version: "0.1.5", reason: "Restored" })
    expect((yield* read)._tag).toBe("Restored")
    expect(Option.getOrThrow(yield* native.inspect(installed, "Magnitude.app"))).toBe(previous)
  })))

  it("never rolls back a committed replacement that later fails verification", () => run(Effect.gen(function* () {
    const { fs, oldPath, write, read, recover, swap, native, installed, replacement } = yield* fixture
    yield* swap
    yield* write("Committed")
    yield* fs.writeFileString(join(oldPath, "version"), "damaged")
    expect(yield* recover.pipe(Effect.isFailure)).toBe(true)
    expect((yield* read)._tag).toBe("Committed")
    expect(Option.getOrThrow(yield* native.inspect(installed, "Magnitude.app"))).toBe(replacement)
  })))

  it.each(["installed-missing", "staging-missing", "installed-substituted", "staging-substituted"])("requires repair for %s", problem => run(Effect.gen(function* () {
    const { fs, oldPath, newPath, write, read, recover } = yield* fixture
    yield* write("ExchangeIntent")
    const path = problem.startsWith("installed") ? oldPath : newPath
    yield* fs.rename(path, `${path}-retained`)
    if (problem.endsWith("substituted")) yield* fs.makeDirectory(path)
    expect(yield* recover.pipe(Effect.isFailure)).toBe(true)
    expect((yield* read)._tag).toBe("ExchangeIntent")
    expect(yield* fs.exists(`${path}-retained`)).toBe(true)
  })))

  it("retains restoration intent when exchange fails before mutation, then resumes", () => run(Effect.gen(function* () {
    const { write, read, recover, swap, native } = yield* fixture
    yield* swap
    yield* write("RestoreIntent")
    expect(yield* recover.pipe(Effect.provideService(MacUpdateFilesystem, { ...native,
      exchange: () => Effect.fail(new MacUpdateFilesystemFailed()) }), Effect.isFailure)).toBe(true)
    expect((yield* read)._tag).toBe("RestoreIntent")
    expect(yield* recover).toMatchObject({ _tag: "Preserved", reason: "Restored" })
  })))

  it("reconciles an error reported after restoration without repeating the exchange", () => run(Effect.gen(function* () {
    const { write, recover, swap, native } = yield* fixture
    yield* swap
    yield* write("RestoreIntent")
    let exchanges = 0
    expect(yield* recover.pipe(Effect.provideService(MacUpdateFilesystem, { ...native,
      exchange: (...args) => Effect.sync(() => exchanges++).pipe(Effect.zipRight(native.exchange(...args)),
        Effect.zipRight(Effect.fail(new MacUpdateFilesystemFailed()))) }))).toMatchObject({ _tag: "Preserved", reason: "Restored" })
    expect(exchanges).toBe(1)
    expect(yield* recover).toMatchObject({ _tag: "Preserved", reason: "Restored" })
    expect(exchanges).toBe(1)
  })))

  it.each(["sync", "record-before", "record-after"])("requires recovery after a %s completion failure", failure => run(Effect.gen(function* () {
    const { write, read, recover, swap, native } = yield* fixture
    yield* write("ExchangeIntent")
    yield* swap
    const broken = { ...native, ...(failure === "sync" ? { sync: () => Effect.fail(new MacUpdateFilesystemFailed()) } : {
      writeRecord: (directory: Parameters<typeof native.writeRecord>[0], bytes: Uint8Array) =>
        (failure === "record-after" ? native.writeRecord(directory, bytes) : Effect.void).pipe(Effect.zipRight(Effect.fail(new MacUpdateFilesystemFailed()))),
    }) }
    expect(yield* recover.pipe(Effect.provideService(MacUpdateFilesystem, broken), Effect.isFailure)).toBe(true)
    expect((yield* read)._tag).toBe(failure === "record-after" ? "Committed" : "ExchangeIntent")
    expect(yield* recover).toEqual({ _tag: "Installed", version: "0.1.6" })
  })))

  it("accepts completed cleanup only after commit", () => run(Effect.gen(function* () {
    const { fs, newPath, write, recover, swap } = yield* fixture
    yield* swap
    yield* write("ExchangeIntent")
    yield* fs.remove(newPath, { recursive: true })
    expect(yield* recover.pipe(Effect.isFailure)).toBe(true)
    yield* write("Committed")
    expect(yield* recover).toEqual({ _tag: "Installed", version: "0.1.6" })
  })))

  it("refuses malformed journals and invalid UTF-8", () => run(Effect.gen(function* () {
    const { native, staging, recover } = yield* fixture
    for (const content of [Buffer.from("{}"), Buffer.from([0xff]), Buffer.from('{"_tag":"Unknown","protocol":1}')]) {
      yield* native.writeRecord(staging, content)
      expect(yield* recover.pipe(Effect.isFailure)).toBe(true)
    }
  })))

  it("cannot restore an old bundle that also fails verification", () => run(Effect.gen(function* () {
    const { fs, oldPath, newPath, write, read, recover, swap, native, installed, replacement } = yield* fixture
    yield* write("ExchangeIntent")
    yield* swap
    yield* fs.writeFileString(join(oldPath, "version"), "damaged replacement")
    yield* fs.writeFileString(join(newPath, "version"), "damaged previous")
    expect(yield* recover.pipe(Effect.isFailure)).toBe(true)
    expect((yield* read)._tag).toBe("ExchangeIntent")
    expect(Option.getOrThrow(yield* native.inspect(installed, "Magnitude.app"))).toBe(replacement)
  })))

  it("binds recovery to both parents and the intended installation name", () => run(Effect.gen(function* () {
    const { native, installed, staging, write, read, recover } = yield* fixture
    yield* write("ExchangeIntent")
    const original = yield* read
    const altered = new TextEncoder().encode(yield* Schema.encode(Schema.parseJson(MacUpdateJournal))({ ...original,
      transaction: { ...original.transaction, installedParent: staging.identity, stagingParent: installed.identity } }))
    yield* native.writeRecord(staging, altered)
    expect(yield* recover.pipe(Effect.isFailure)).toBe(true)
    yield* write("ExchangeIntent")
    expect(yield* recoverMacUpdateTransaction(installed, "Different.app", staging).pipe(
      Effect.provideService(MacBundleVerifier, { verify: () => Effect.void }), Effect.isFailure)).toBe(true)
    expect((yield* read)._tag).toBe("ExchangeIntent")
  })))

  const matrix = [
    ["ExchangeIntent", "OldAndNew", "Preserved"], ["ExchangeIntent", "NewAndOld", "Installed"],
    ["ExchangeIntent", "OldOnly", "Repair"], ["ExchangeIntent", "NewOnly", "Repair"],
    ["Committed", "OldAndNew", "Repair"], ["Committed", "NewAndOld", "Installed"],
    ["Committed", "OldOnly", "Repair"], ["Committed", "NewOnly", "Installed"],
    ["RestoreIntent", "OldAndNew", "Preserved"], ["RestoreIntent", "NewAndOld", "Preserved"],
    ["RestoreIntent", "OldOnly", "Repair"], ["RestoreIntent", "NewOnly", "Repair"],
    ["Restored", "OldAndNew", "Preserved"], ["Restored", "NewAndOld", "Repair"],
    ["Restored", "OldOnly", "Preserved"], ["Restored", "NewOnly", "Repair"],
    ["Abandoned", "OldAndNew", "Preserved"], ["Abandoned", "NewAndOld", "Repair"],
    ["Abandoned", "OldOnly", "Preserved"], ["Abandoned", "NewOnly", "Repair"],
  ] as const
  it.each(matrix)("reconciles %s with %s as %s", (tag, layout, outcome) => run(Effect.gen(function* () {
    const { fs, newPath, write, recover, swap } = yield* fixture
    if (layout.startsWith("New")) yield* swap
    if (layout.endsWith("Only")) yield* fs.remove(newPath, { recursive: true })
    yield* write(tag)
    const result = yield* recover.pipe(Effect.either)
    if (outcome === "Repair") {
      expect(result._tag).toBe("Left")
      if (result._tag === "Left") expect(result.left._tag).toBe("MacUpdateRepairRequired")
    } else {
      expect(result._tag).toBe("Right")
      if (result._tag === "Right") expect(result.right._tag).toBe(outcome)
    }
  })))
})
