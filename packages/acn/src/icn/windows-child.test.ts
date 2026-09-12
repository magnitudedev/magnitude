import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnChildLaunch, IcnChildSpawner } from "@magnitudedev/icn"
import { Duration } from "effect"
import { WindowsPrivatePipes, WindowsPipeFailed, WindowsProcessId, windowsJobOwnerLayer } from "@magnitudedev/utils/windows-native"
import { WindowsIcnChildSpawner } from "./windows-child"

const launch = new IcnChildLaunch({ executable: "C:\\Magnitude\\inference.exe", arguments: ["serve", "--exit-on-stdin-eof"], environment: { MAGNITUDE_ICN_AUTH_TOKEN: "fixture-token" }, gracefulShutdownTimeout: Duration.seconds(1), forceShutdownTimeout: Duration.seconds(1) })
const fixture = (failAccept = false, cooperative = true, unprovenRetirement = false) => {
  const observed = { spawned: 0, terminated: 0, closed: 0, pipesClosed: 0, environment: "", streams: [] as string[], exited: false, exitCode: 0, shutdownFrames: [] as string[] }
  const jobs = windowsJobOwnerLayer({
    spawnOwnedProcess: () => { throw new Error("Inference requires separate streams") },
    spawnOwnedProcessWithPipes: (_exe, _cmd, env, ...streams) => { observed.spawned++; observed.environment = env; observed.streams = streams; return {} },
    ownedProcessIdentity: () => ({ pid: 42, creationTime: "1234567890abcdef" }),
    ownedProcessActiveCount: () => unprovenRetirement ? 1 : observed.exited ? 0 : 1,
    ownedProcessExit: () => observed.exited ? observed.exitCode : null,
    terminateOwnedProcess: () => { observed.terminated++; if (!observed.exited) observed.exitCode = 1; observed.exited = true },
    closeOwnedProcess: () => { observed.closed++ },
  })
  const pipes = Layer.succeed(WindowsPrivatePipes, { bind: name => Effect.acquireRelease(Effect.sync(() => {
    let read = false
    return {
      accept: failAccept ? Effect.fail(new WindowsPipeFailed({ message: "fixture admission failure", win32Code: Option.none() })) : Effect.succeed(WindowsProcessId.make(7)),
      read: Effect.sync(() => { if (read) return new Uint8Array(); read = true; return Buffer.from(name.endsWith("stdout") ? "startup record" : "diagnostic") }),
      write: (bytes: Uint8Array) => failAccept ? Effect.fail(new WindowsPipeFailed({ message: "not admitted", win32Code: Option.none() })) : Effect.sync(() => { observed.shutdownFrames.push(Buffer.from(bytes).toString()); if (cooperative) observed.exited = true }), close: Effect.void,
    }
  }), () => Effect.sync(() => { observed.pipesClosed++ })) })
  return { observed, layer: WindowsIcnChildSpawner.pipe(Layer.provide(Layer.merge(jobs, pipes))) }
}
describe("Windows ICN child (simulated native boundaries)", () => {
  it("keeps inherited lifetime input open, separates records and diagnostics, and retires one job", async () => {
    const test = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const children = yield* IcnChildSpawner
      const child = yield* children.spawn(launch)
      expect(child.pid).toBe(42)
      expect(yield* child.stdout.pipe(Stream.decodeText(), Stream.runFold("", (text, next) => text + next))).toBe("startup record")
      expect(yield* child.stderr.pipe(Stream.decodeText(), Stream.runFold("", (text, next) => text + next))).toBe("diagnostic")
      expect(test.observed.pipesClosed).toBe(0)
      expect(test.observed.environment).toContain("MAGNITUDE_ICN_AUTH_TOKEN=fixture-token\0")
      expect(new Set(test.observed.streams).size).toBe(3)
      yield* child.terminate
      yield* child.terminate
      expect(yield* child.exitCode).toBe(0)
    })).pipe(Effect.provide(test.layer)))
    expect(test.observed).toMatchObject({ spawned: 1, terminated: 1, closed: 1, pipesClosed: 3 })
    expect(test.observed.shutdownFrames).toEqual(['{"type":"shutdown"}\n'])
  })
  it("failed inherited-stream admission retires the child and releases all pipes", async () => {
    const test = fixture(true)
    const result = await Effect.runPromise(Effect.either(Effect.scoped(Effect.flatMap(IcnChildSpawner, children => children.spawn(launch))).pipe(Effect.provide(test.layer))))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("IcnChildAcquisitionFailed")
    expect(test.observed).toMatchObject({ spawned: 1, terminated: 1, closed: 1, pipesClosed: 3 })
  })
  it("escalates after a missed grace deadline while keeping shutdown single-flight", async () => {
    const test = fixture(false, false)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const children = yield* IcnChildSpawner
      const child = yield* children.spawn(new IcnChildLaunch({ ...launch, gracefulShutdownTimeout: Duration.millis(30) }))
      yield* Effect.all([child.terminate, child.terminate], { concurrency: "unbounded" })
      expect(yield* child.exitCode).toBe(1)
    })).pipe(Effect.provide(test.layer)))
    expect(test.observed.shutdownFrames).toHaveLength(1)
    expect(test.observed).toMatchObject({ terminated: 1, closed: 1, pipesClosed: 3 })
  })
  it("bounds forced retirement and retains the job until ACN scope cleanup when descendants remain", async () => {
    const test = fixture(false, true, true)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const children = yield* IcnChildSpawner
      const child = yield* children.spawn(new IcnChildLaunch({ ...launch, forceShutdownTimeout: Duration.millis(30) }))
      const result = yield* Effect.either(child.terminate)
      expect(result._tag).toBe("Left")
      if (result._tag === "Left") expect(result.left._tag).toBe("IcnChildRetirementFailed")
      expect(test.observed.closed).toBe(0)
    })).pipe(Effect.provide(test.layer)))
    expect(test.observed.closed).toBe(1)
  })
})
