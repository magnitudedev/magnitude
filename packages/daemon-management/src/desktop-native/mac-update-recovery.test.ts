import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Deferred, Effect, Exit, Fiber, Option, Schema, Scope } from "effect"
import { spawn } from "node:child_process"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { MacUpdateFilesystem, MacUpdateFilesystemFailed, nativeMacUpdateFilesystem } from "./mac-update-filesystem"
import { MacBundleVerificationFailed, MacBundleVerifier } from "./mac-update-validation"
import { MacUpdateJournal, recoverMacUpdateTransaction, retireMacUpdateBundle } from "./mac-update-recovery"
import { exchangeMacUpdate } from "./mac-update-transaction"

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
  const apply = exchangeMacUpdate(installed, "Magnitude.app", staging, { previous: "0.1.5", replacement: "0.1.6", architecture: "arm64" }).pipe(
    Effect.provideService(MacBundleVerifier, verifier))
  const retire = retireMacUpdateBundle(installed, "Magnitude.app", staging).pipe(Effect.provideService(MacBundleVerifier, verifier))
  return { fs, native, root, installed, staging, oldPath, newPath, previous, replacement, write, read, recover, swap, apply, retire }
})

describe.skipIf(process.platform !== "darwin")("macOS prepared bundle exchange", () => {
  it("verifies, synchronizes, journals and installs through recovery", () => run(Effect.gen(function* () {
    const { apply, read, native } = yield* fixture
    const events: string[] = []
    const traced = { ...native,
      syncTree: (...args: Parameters<typeof native.syncTree>) => native.syncTree(...args).pipe(Effect.tap(() => Effect.sync(() => events.push("tree")))),
      writeRecord: (...args: Parameters<typeof native.writeRecord>) => native.writeRecord(...args).pipe(Effect.tap(() => Effect.sync(() => events.push("record")))),
      exchange: (...args: Parameters<typeof native.exchange>) => native.exchange(...args).pipe(Effect.tap(() => Effect.sync(() => events.push("exchange")))),
    }
    expect(yield* apply.pipe(Effect.provideService(MacUpdateFilesystem, traced))).toEqual({ _tag: "Installed", version: "0.1.6" })
    expect(events).toEqual(["tree", "record", "exchange", "record"])
    expect((yield* read)._tag).toBe("Committed")
  })))

  it.each(["previous", "replacement"])("refuses an invalid %s bundle before journal publication", invalid => run(Effect.gen(function* () {
    const { apply, fs, native, staging, oldPath, newPath } = yield* fixture
    yield* fs.writeFileString(join(invalid === "previous" ? oldPath : newPath, "version"), "damaged")
    const result = yield* apply.pipe(Effect.either)
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe(invalid === "previous" ? "MacUpdateRepairRequired" : "MacUpdatePreparationFailed")
    expect(Option.isNone(yield* native.readRecord(staging))).toBe(true)
  })))

  it("does not publish intent when staged synchronization fails", () => run(Effect.gen(function* () {
    const { apply, native, staging } = yield* fixture
    expect(yield* apply.pipe(Effect.provideService(MacUpdateFilesystem, { ...native,
      syncTree: () => Effect.fail(new MacUpdateFilesystemFailed()) }), Effect.isFailure)).toBe(true)
    expect(Option.isNone(yield* native.readRecord(staging))).toBe(true)
  })))

  it("does not replace an existing transaction journal", () => run(Effect.gen(function* () {
    const { apply, write, read } = yield* fixture
    yield* write("ExchangeIntent")
    const before = yield* read
    expect(yield* apply.pipe(Effect.isFailure)).toBe(true)
    expect(yield* read).toEqual(before)
  })))

  it.each([false, true])("reconciles exchange failure after mutation = %s", mutated => run(Effect.gen(function* () {
    const { apply, native, read } = yield* fixture
    const result = yield* apply.pipe(Effect.provideService(MacUpdateFilesystem, { ...native,
      exchange: (...args) => (mutated ? native.exchange(...args) : Effect.void).pipe(Effect.zipRight(Effect.fail(new MacUpdateFilesystemFailed()))) }))
    expect(result._tag).toBe(mutated ? "Installed" : "Preserved")
    expect((yield* read)._tag).toBe(mutated ? "Committed" : "Abandoned")
  })))

  it("cancels before intent without authorizing exchange", () => run(Effect.gen(function* () {
    const { apply, native, staging } = yield* fixture
    const waiting = yield* Deferred.make<void>()
    const worker = yield* apply.pipe(Effect.provideService(MacUpdateFilesystem, { ...native,
      syncTree: () => Deferred.succeed(waiting, undefined).pipe(Effect.zipRight(Effect.never)) }), Effect.forkScoped)
    yield* Deferred.await(waiting)
    expect(Exit.isInterrupted(yield* Fiber.interrupt(worker))).toBe(true)
    expect(Option.isNone(yield* native.readRecord(staging))).toBe(true)
  })))

  it("finishes reconciliation when cancellation arrives after intent", () => run(Effect.gen(function* () {
    const { apply, native, read } = yield* fixture
    const published = yield* Deferred.make<void>()
    const resume = yield* Deferred.make<void>()
    const worker = yield* apply.pipe(Effect.provideService(MacUpdateFilesystem, { ...native,
      writeRecord: (...args) => native.writeRecord(...args).pipe(Effect.zipRight(Deferred.succeed(published, undefined)), Effect.zipRight(Deferred.await(resume))),
    }), Effect.forkScoped)
    yield* Deferred.await(published)
    yield* Fiber.interruptFork(worker)
    yield* Deferred.succeed(resume, undefined)
    expect(Exit.isInterrupted(yield* Fiber.await(worker))).toBe(true)
    expect((yield* read)._tag).toBe("Committed")
  })))
})

describe.skipIf(process.platform !== "darwin")("macOS transaction recovery on native filesystems", () => {
  it.each(["BeforeExchange", "AfterExchange", "AfterCommit", "BeforeRestore", "AfterRestore", "PartialCleanup", "AfterTreeRemoval", "AfterReceiptRemoval"])("recovers after actual process loss at %s", phase => run(Effect.gen(function* () {
    const { root, staging, write, recover, retire, fs, oldPath, newPath } = yield* fixture
    yield* write("ExchangeIntent")
    yield* Effect.async<void>(resume => {
      const child = spawn(process.execPath, [fileURLToPath(new URL("./fixtures/mac-update-recovery-crash.ts", import.meta.url)), root, staging.path, phase], { stdio: ["ignore", "ignore", "inherit"] })
      child.once("error", error => resume(Effect.die(error)))
      child.once("exit", (_code, signal) => resume(Effect.sync(() => expect(signal).toBe("SIGKILL"))))
      return Effect.sync(() => { child.kill("SIGKILL") })
    })
    const expected = phase === "AfterReceiptRemoval" ? { _tag: "NoTransaction" } : phase === "BeforeExchange" ? { _tag: "Preserved", reason: "Interrupted" }
      : phase.endsWith("Restore") ? { _tag: "Preserved", reason: "Restored" } : { _tag: "Installed", version: "0.1.6" }
    expect(yield* recover).toMatchObject(expected)
    expect(yield* recover).toMatchObject(expected)
    if (["PartialCleanup", "AfterTreeRemoval", "AfterReceiptRemoval"].includes(phase)) {
      expect(yield* fs.readFileString(join(oldPath, "version"))).toBe("0.1.6")
      yield* retire
      expect(yield* fs.exists(newPath)).toBe(false)
      expect(yield* recover).toEqual({ _tag: "NoTransaction" })
    }
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

describe.skipIf(process.platform !== "darwin")("macOS completed transaction cleanup", () => {
  it.each(["Committed", "Restored", "Abandoned"] as const)("retires only displaced contents for %s and replays without exchange", tag => run(Effect.gen(function* () {
    const { fs, newPath, oldPath, write, read, swap, retire, recover } = yield* fixture
    if (tag === "Committed") yield* swap
    yield* write(tag)
    const expected = yield* recover
    expect(yield* retire).toEqual(expected)
    expect(yield* fs.exists(newPath)).toBe(false)
    expect(yield* fs.readFileString(join(oldPath, "version"))).toBe(tag === "Committed" ? "0.1.6" : "0.1.5")
    expect(yield* retire).toEqual({ _tag: "NoTransaction" })
    expect(yield* recover).toEqual({ _tag: "NoTransaction" })
  })))

  it.each(["ExchangeIntent", "RestoreIntent"] as const)("refuses cleanup of %s", tag => run(Effect.gen(function* () {
    const { fs, newPath, write, read, retire } = yield* fixture
    yield* write(tag)
    expect(yield* retire.pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readFileString(join(newPath, "version"))).toBe("0.1.6")
    expect((yield* read)._tag).toBe(tag)
  })))

  it("leaves unjournaled contents untouched", () => run(Effect.gen(function* () {
    const { fs, newPath, retire } = yield* fixture
    expect(yield* retire).toEqual({ _tag: "NoTransaction" })
    expect(yield* fs.readFileString(join(newPath, "version"))).toBe("0.1.6")
  })))

  it("requires verified installed contents before deleting the displaced bundle", () => run(Effect.gen(function* () {
    const { fs, oldPath, newPath, write, retire, swap } = yield* fixture
    yield* swap
    yield* write("Committed")
    yield* fs.writeFileString(join(oldPath, "version"), "damaged")
    expect(yield* retire.pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.readFileString(join(newPath, "version"))).toBe("0.1.5")
  })))

  it("retains the terminal receipt after partial cleanup and retries only deletion", () => run(Effect.gen(function* () {
    const { fs, native, newPath, write, read, swap, retire, recover } = yield* fixture
    yield* swap
    yield* write("Committed")
    const failed = yield* retire.pipe(Effect.provideService(MacUpdateFilesystem, { ...native,
      removeTree: () => fs.remove(join(newPath, "version")).pipe(Effect.orDie,
        Effect.zipRight(Effect.fail(new MacUpdateFilesystemFailed()))),
    }), Effect.either)
    expect(failed._tag).toBe("Left")
    if (failed._tag === "Left") expect(failed.left._tag).toBe("MacUpdateCleanupFailed")
    expect((yield* read)._tag).toBe("Committed")
    expect(yield* recover).toEqual({ _tag: "Installed", version: "0.1.6" })
    expect(yield* retire).toEqual({ _tag: "Installed", version: "0.1.6" })
    expect(yield* fs.exists(newPath)).toBe(false)
  })))
})
