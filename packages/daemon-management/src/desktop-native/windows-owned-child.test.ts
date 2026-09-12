import { Deferred, Effect, Layer, Option, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { windowsJobOwnerLayer } from "@magnitudedev/utils/windows-native"
import { WindowsPrivatePipes, WindowsPipeFailed } from "@magnitudedev/utils/windows-native"
import { WindowsProcessId } from "@magnitudedev/utils/windows-native"
import { makeWindowsOwnedChildSpawner } from "./windows-owned-child"

const fixture = (peer = 42, earlyExit: number | null = null) => {
  const observed = { spawned: 0, retired: 0, closed: 0, environment: "", writes: [] as string[], pipesClosed: new Set<string>() }
  let exit = earlyExit
  const jobs = windowsJobOwnerLayer({
    spawnOwnedProcess: (_executable, _command, environment) => { observed.spawned++; observed.environment = environment; return {} },
    spawnOwnedProcessWithPipes: () => { throw new Error("Service diagnostics must remain merged") },
    ownedProcessIdentity: () => ({ pid: 42, creationTime: "1234567890abcdef" }),
    ownedProcessActiveCount: () => exit === null ? 1 : 0,
    ownedProcessExit: () => exit,
    terminateOwnedProcess: () => { observed.retired++; exit = 1 },
    closeOwnedProcess: () => { observed.closed++ },
  })
  const pipes = Layer.succeed(WindowsPrivatePipes, WindowsPrivatePipes.of({ bind: name => Effect.gen(function* () {
    const ended = yield* Deferred.make<Uint8Array, WindowsPipeFailed>()
    const sent = yield* Ref.make(false)
    const close = Effect.sync(() => { observed.pipesClosed.add(name) }).pipe(Effect.zipRight(Deferred.fail(ended, new WindowsPipeFailed({ message: "closed", win32Code: Option.none() }))), Effect.asVoid)
    yield* Effect.addFinalizer(() => close)
    return {
      accept: earlyExit !== null && name.includes("child-") ? Effect.never : Effect.succeed(WindowsProcessId.make(name.includes("child-") ? peer : process.pid)),
      read: Ref.getAndSet(sent, true).pipe(Effect.flatMap(already => already ? Deferred.await(ended) : Effect.succeed(Buffer.from(name.includes("child-") ? '{"_tag":"Booted","pid":42}\n' : 'service diagnostic\n')))),
      write: bytes => Effect.sync(() => { observed.writes.push(Buffer.from(bytes).toString()) }), close,
    }
  }) }))
  return { observed, layer: Layer.merge(jobs, pipes) }
}
const command = { executable: "C:\\Magnitude\\magnitude-service.exe", arguments: [], environment: { MAGNITUDE_OWNER_PIPE: "caller-value" } }
describe("Windows owned service composition (simulated native boundary)", () => {
  it("fences the control peer, retains diagnostics, encodes commands and retires the owned job", async () => {
    const test = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const spawner = yield* makeWindowsOwnedChildSpawner
      const child = yield* spawner.spawn(command)
      expect(child.identity.pid).toBe(42)
      expect(test.observed.environment).not.toContain("caller-value")
      expect(test.observed.environment).toContain("MAGNITUDE_OWNER_PIPE=\\\\.\\pipe\\magnitude-child-")
      const boot = yield* child.events.pipe(Stream.take(1), Stream.runHead)
      expect(Option.getOrThrow(boot)).toEqual({ _tag: "Booted", pid: 42 })
      expect(yield* child.diagnosticTail).toContain("service diagnostic")
      yield* child.send({ _tag: "Start" })
      expect(JSON.parse(test.observed.writes[0]!)).toEqual({ _tag: "Start" })
      yield* child.stop
      expect(yield* child.exit).toBe(1)
    })).pipe(Effect.provide(test.layer)))
    expect(test.observed.retired).toBe(1)
    expect(test.observed.closed).toBe(1)
    expect(test.observed.pipesClosed.size).toBe(2)
  })

  it("rejects a different native pipe client before handing control to the supervisor", async () => {
    const test = fixture(99)
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const spawner = yield* makeWindowsOwnedChildSpawner
      return yield* Effect.either(spawner.spawn(command))
    })).pipe(Effect.provide(test.layer)))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("OwnedChildSpawnFailed")
    expect(test.observed.writes).toEqual([])
    expect(test.observed.retired).toBe(1)
    expect(test.observed.closed).toBe(1)
  })

  it("reports root exit during admission promptly and cleans the job", async () => {
    const test = fixture(42, 3)
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const spawner = yield* makeWindowsOwnedChildSpawner
      return yield* Effect.either(spawner.spawn(command))
    })).pipe(Effect.provide(test.layer), Effect.timeout("1 second")))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain("exit code 3")
    expect(test.observed.closed).toBe(1)
    expect(test.observed.pipesClosed.size).toBe(2)
  })
})
