import { Deferred, Effect, Fiber, Layer, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { encodeWindowsCommand } from "./windows-command"
import { WindowsJobOwner, windowsJobOwnerLayer, type WindowsJobBindings, type WindowsOwnedJob } from "./windows-job"
import { WindowsPipeName } from "./windows-pipe"

const command = Effect.runSync(encodeWindowsCommand({ executable: "C:\\Magnitude\\service.exe", arguments: [], environment: {} }))
const output = { _tag: "Diagnostics" as const, output: WindowsPipeName.make("\\\\.\\pipe\\magnitude-job-output-fixture") }
const fixture = (overrides: Partial<WindowsJobBindings> = {}) => {
  const observed = { spawned: 0, closed: 0, terminated: 0, exit: null as number | null, active: 1 }
  const native: WindowsJobBindings = {
    spawnOwnedProcess: () => ({ id: ++observed.spawned }),
    spawnOwnedProcessWithPipes: () => ({ id: ++observed.spawned }),
    ownedProcessIdentity: () => ({ pid: 42, creationTime: "1234567890abcdef" }),
    ownedProcessActiveCount: () => observed.active,
    ownedProcessExit: () => observed.exit,
    terminateOwnedProcess: () => { observed.terminated++; observed.active = 0; observed.exit = 1 },
    closeOwnedProcess: () => { observed.closed++ }, ...overrides,
  }
  return { observed, layer: windowsJobOwnerLayer(native) }
}
describe("Windows job ownership (simulated native boundary)", () => {
  it("retains the same ownership guarantees with separate inherited streams", async () => {
    const streams = { _tag: "Separate" as const, input: WindowsPipeName.make("\\\\.\\pipe\\magnitude-input"), output: output.output, error: WindowsPipeName.make("\\\\.\\pipe\\magnitude-error") }
    let supplied: unknown[] = []
    const test = fixture({ spawnOwnedProcess: () => { throw new Error("Must use separate streams") }, spawnOwnedProcessWithPipes: (...args) => { supplied = args; return {} } })
    await Effect.runPromise(Effect.gen(function* () {
      const owner = yield* WindowsJobOwner
      const job = yield* owner.spawn(command, streams)
      expect(supplied).toEqual([command.executable, command.commandLine, command.environment, streams.input, streams.output, streams.error])
      expect((yield* Effect.either(owner.spawn(command, output)))._tag).toBe("Left")
      yield* job.retire("10 seconds")
      expect(test.observed.closed).toBe(1)
    }).pipe(Effect.provide(test.layer)))
  })
  it("waits for descendants after root exit, caches exit and allows another job only after retirement", async () => {
    const counts = [2, 1, 0]
    const test = fixture({ ownedProcessActiveCount: () => counts.shift() ?? 0, ownedProcessExit: () => 0, terminateOwnedProcess: () => {} })
    await Effect.runPromise(Effect.gen(function* () {
      const owner = yield* WindowsJobOwner
      const first = yield* owner.spawn(command, output)
      expect((yield* first.identity).pid).toBe(42)
      expect(yield* first.exit).toBe(0)
      expect(test.observed.closed).toBe(0)
      expect((yield* Effect.either(owner.spawn(command, output)))._tag).toBe("Left")
      yield* first.retire("10 seconds")
      expect(counts).toEqual([])
      expect(test.observed.closed).toBe(1)
      const second = yield* owner.spawn(command, output)
      yield* first.retire("10 seconds")
      expect(test.observed.closed).toBe(1)
      expect(yield* first.exit).toBe(0)
      yield* second.retire("10 seconds")
    }).pipe(Effect.provide(test.layer)))
    expect(test.observed.closed).toBe(2)
  })

  it("keeps native ownership through failed cleanup and retries using the same handle", async () => {
    let fail = true
    const test = fixture({ ownedProcessActiveCount: () => { if (fail) throw { win32Code: 5 }; return 0 } })
    await Effect.runPromise(Effect.gen(function* () {
      const owner = yield* WindowsJobOwner
      const job = yield* owner.spawn(command, output)
      expect((yield* Effect.either(job.retire("10 seconds")))._tag).toBe("Left")
      expect(test.observed.closed).toBe(0)
      expect((yield* job.identity).creationTime).toBe("1234567890abcdef")
      expect((yield* Effect.either(owner.spawn(command, output)))._tag).toBe("Left")
      expect(test.observed.spawned).toBe(1)
      fail = false
      yield* job.retire("10 seconds")
      expect(test.observed.closed).toBe(1)
    }).pipe(Effect.provide(test.layer)))
    expect(test.observed.closed).toBe(1)
  })

  it("retains ownership when retirement is interrupted and releases it only at application scope exit", async () => {
    const test = fixture({ terminateOwnedProcess: () => {} })
    let captured!: WindowsOwnedJob
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const owner = yield* WindowsJobOwner
      captured = yield* owner.spawn(command, output)
      const attempt = yield* Effect.forkScoped(captured.retire("10 seconds"))
      yield* Effect.yieldNow()
      yield* Fiber.interrupt(attempt)
      expect(test.observed.closed).toBe(0)
      expect((yield* Effect.either(owner.spawn(command, output)))._tag).toBe("Left")
    })).pipe(Effect.provide(test.layer)))
    expect(test.observed.closed).toBe(1)
    expect((await Effect.runPromise(Effect.either(captured.exit)))._tag).toBe("Left")
  })

  it("wakes an exit observer with the retained result after retirement closes the handle", async () => {
    const test = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const owner = yield* WindowsJobOwner; const job = yield* owner.spawn(command, output)
      const exit = yield* Effect.forkScoped(job.exit)
      yield* job.retire("10 seconds")
      expect(yield* Fiber.join(exit)).toBe(1)
    })).pipe(Effect.provide(test.layer)))
  })

  it("does not discard a process when identity observation is invalid", async () => {
    const test = fixture({ ownedProcessIdentity: () => ({ pid: 0, creationTime: "wrong" }) })
    await Effect.runPromise(Effect.gen(function* () {
      const owner = yield* WindowsJobOwner; const job = yield* owner.spawn(command, output)
      expect((yield* Effect.either(job.identity))._tag).toBe("Left")
      expect(test.observed.closed).toBe(0)
      yield* job.retire("10 seconds")
    }).pipe(Effect.provide(test.layer)))
    expect(test.observed.closed).toBe(1)
  })

  it("retains handles after the retirement deadline and forbids replacement", async () => {
    const test = fixture({ terminateOwnedProcess: () => {} })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const owner = yield* WindowsJobOwner; const job = yield* owner.spawn(command, output)
      const started = yield* Deferred.make<void>()
      const attempt = yield* Effect.forkScoped(Deferred.succeed(started, undefined).pipe(Effect.zipRight(Effect.either(job.retire("10 seconds")))))
      yield* Deferred.await(started)
      yield* TestClock.adjust("11 seconds")
      expect((yield* Fiber.join(attempt))._tag).toBe("Left")
      expect(test.observed.closed).toBe(0)
      expect((yield* Effect.either(owner.spawn(command, output)))._tag).toBe("Left")
    })).pipe(Effect.provide(Layer.merge(test.layer, TestContext.TestContext))))
    expect(test.observed.closed).toBe(1)
  })
})
