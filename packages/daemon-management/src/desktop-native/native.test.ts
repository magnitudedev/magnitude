import { spawn, type ChildProcess } from "node:child_process"
import { chmodSync, existsSync, linkSync, mkdtempSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs"
import { createRequire } from "node:module"
import { tmpdir } from "node:os"
import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Option, Schema } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { NativeHost, nativeHostLayer } from "./index"

const addon = fileURLToPath(new URL(`../../dist/native/${process.platform}-${process.arch}/desktop-host.node`, import.meta.url))
const fixture = fileURLToPath(new URL("./fixtures/process.cjs", import.meta.url))
const processes: ChildProcess[] = []
const directories: string[] = []
const strayPids: number[] = []
const live = (pid: number) => { try { process.kill(pid, 0); return true } catch { return false } }

class FixtureFailed extends Schema.TaggedError<FixtureFailed>()("FixtureFailed", { message: Schema.String }) {}

const launch = (mode: string, argument = "") => Effect.async<{ child: ChildProcess; line: string }, FixtureFailed>(resume => {
  const child = spawn(process.execPath, [fixture, addon, mode, argument], { stdio: ["pipe", "pipe", "pipe"] })
  processes.push(child)
  let output = ""
  let error = ""
  child.stderr!.on("data", chunk => { error += chunk.toString() })
  child.stdout!.on("data", chunk => {
    output += chunk.toString()
    if (output.includes("\n")) resume(Effect.succeed({ child, line: output.split("\n")[0]! }))
  })
  child.once("error", error => resume(Effect.fail(new FixtureFailed({ message: error.message }))))
  child.once("exit", () => { if (!output.includes("\n")) resume(Effect.fail(new FixtureFailed({ message: error || "No fixture output" }))) })
}).pipe(Effect.timeout("5 seconds"))

// Native process exit is observed, not inferred from the leader's exit event.
const absent = (pid: number) => Effect.gen(function* () {
  for (let i = 0; i < 100; i++) {
    if (!live(pid)) return
    yield* Effect.sleep("20 millis")
  }
  return yield* new FixtureFailed({ message: `Process ${pid} survived cleanup` })
})
const directory = () => { const dir = mkdtempSync(resolve(tmpdir(), "magnitude-native-")); directories.push(dir); return dir }

afterEach(async () => {
  for (const child of processes.splice(0)) if (child.exitCode === null) child.kill("SIGKILL")
  for (const pid of strayPids.splice(0)) { try { process.kill(pid, "SIGKILL") } catch {} }
  for (const dir of directories.splice(0)) rmSync(dir, { recursive: true, force: true })
})

describe.skipIf(process.platform === "win32")("native desktop ownership (real processes)", () => {
  it("requires the compiled acceptance artifact", () => expect(existsSync(addon)).toBe(true))

  it("refuses a live owner without takeover and reacquires the same file after SIGKILL", async () => {
    const path = resolve(directory(), "owner.lock")
    const owner = await Effect.runPromise(launch("owner", path))
    expect(owner.line).toBe("owned")
    const inode = statSync(path).ino
    const contender = await Effect.runPromise(launch("owner", path))
    expect(contender.line).toBe("contended")
    expect(live(owner.child.pid!)).toBe(true)
    owner.child.kill("SIGKILL")
    await Effect.runPromise(absent(owner.child.pid!))
    const next = await Effect.runPromise(launch("owner", path))
    expect(next.line).toBe("owned")
    expect(statSync(path).ino).toBe(inode)
  })

  it("releases ownership with the Effect scope", async () => {
    const path = resolve(directory(), "owner.lock")
    await Effect.runPromise(Effect.gen(function* () {
      const host = yield* NativeHost
      yield* Effect.scoped(Effect.gen(function* () {
        expect(Option.isSome(yield* host.acquireOwnership(path))).toBe(true)
        expect(Option.isNone(yield* host.acquireOwnership(path))).toBe(true)
      }))
      yield* Effect.scoped(Effect.gen(function* () {
        expect(Option.isSome(yield* host.acquireOwnership(path))).toBe(true)
      }))
    }).pipe(Effect.provide(nativeHostLayer(addon))))
  })

  it("does not retain the lock in an exec'd descendant", async () => {
    const path = resolve(directory(), "owner.lock")
    const owner = await Effect.runPromise(launch("owner-child", path))
    const pids = JSON.parse(owner.line)
    strayPids.push(pids.worker)
    owner.child.kill("SIGKILL")
    await Effect.runPromise(absent(pids.owner))
    expect(live(pids.worker)).toBe(true)
    expect((await Effect.runPromise(launch("owner", path))).line).toBe("owned")
  })

  it.each(["normal", "blocked"])("kills real child and worker when the owner dies (%s JavaScript)", async mode => {
    const parent = await Effect.runPromise(launch("parent", mode))
    const pids = JSON.parse(parent.line)
    strayPids.push(pids.child, pids.worker)
    expect(live(pids.worker)).toBe(true)
    parent.child.kill("SIGKILL")
    await Effect.runPromise(Effect.all([absent(pids.child), absent(pids.worker)], { concurrency: "unbounded" }))
  })

  it("fails on unsafe lock files rather than treating them as contention", async () => {
    const dir = directory()
    const unsafe = resolve(dir, "unsafe")
    const link = resolve(dir, "link")
    writeFileSync(unsafe, "", { mode: 0o644 })
    chmodSync(unsafe, 0o644)
    symlinkSync(unsafe, link)
    await Effect.runPromise(Effect.gen(function* () {
      const host = yield* NativeHost
      for (const path of [unsafe, link, dir]) {
        const outcome = yield* Effect.scoped(host.acquireOwnership(path)).pipe(Effect.either)
        expect(outcome._tag).toBe("Left")
      }
    }).pipe(Effect.provide(nativeHostLayer(addon))))
  })

  it("rejects NUL-suffixed paths without creating or locking the truncated path", async () => {
    const path = resolve(directory(), "must-not-exist.lock")
    await Effect.runPromise(Effect.gen(function* () {
      const host = yield* NativeHost
      const outcome = yield* Effect.scoped(host.acquireOwnership(path + "\0ignored")).pipe(Effect.either)
      expect(outcome._tag).toBe("Left")
      expect(existsSync(path)).toBe(false)
    }).pipe(Effect.provide(nativeHostLayer(addon))))
  })

  it("rejects hard-linked lock aliases", async () => {
    const dir = directory(); const first = resolve(dir, "first.lock"); const second = resolve(dir, "second.lock")
    writeFileSync(first, "", { mode: 0o600 }); linkSync(first, second)
    await Effect.runPromise(Effect.gen(function* () {
      const host = yield* NativeHost
      for (const path of [first, second]) expect((yield* Effect.scoped(Effect.either(host.acquireOwnership(path))))._tag).toBe("Left")
    }).pipe(Effect.provide(nativeHostLayer(addon))))
  })

  it("rejects forged release capabilities and permits repeated release of the actual handle", () => {
    const native = createRequire(import.meta.url)(addon) as { acquireLock: (path: string) => object; releaseLock: (handle: object) => void }
    const path = resolve(directory(), "owner.lock"); const lock = native.acquireLock(path)
    expect(() => native.releaseLock({})).toThrow("Invalid ownership lock")
    expect(() => native.releaseLock(Object.create(lock))).toThrow("Invalid ownership lock")
    expect(native.acquireLock(path)).toBeNull()
    native.releaseLock(lock); native.releaseLock(lock)
    const next = native.acquireLock(path)
    expect(next).not.toBeNull(); native.releaseLock(next)
  })
})
