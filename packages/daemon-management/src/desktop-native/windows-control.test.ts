import { Deferred, Effect, Fiber, Schedule, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ApplicationSnapshot } from "./application-control"
import { LoginStartupFailed } from "@magnitudedev/sdk/desktop-host"
import { serveWindowsApplicationControl } from "./windows-control"
import { WindowsPipeName, windowsPrivatePipesLayer, type WindowsPipeBindings } from "@magnitudedev/utils/windows-native"

const pending = <A>() => {
  let resolve!: (value: A) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<A>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
const fixture = () => {
  const instances: Array<{ first: boolean; accept: ReturnType<typeof pending<number>>; input: ReturnType<typeof pending<Uint8Array>>; output: Buffer[]; reads: Array<ReturnType<typeof pending<Uint8Array>>>; closed: boolean }> = []
  const native: WindowsPipeBindings = {
    createPrivatePipe: (_name, first) => {
      const pipe = { first, accept: pending<number>(), input: pending<Uint8Array>(), output: [] as Buffer[], reads: [] as Array<ReturnType<typeof pending<Uint8Array>>>, closed: false }
      // The fake transport has no kernel waiting on these until the adapter issues an operation.
      void pipe.accept.promise.catch(() => {}); void pipe.input.promise.catch(() => {})
      instances.push(pipe)
      return pipe
    },
    acceptPrivatePipe: pipe => (pipe as typeof instances[number]).accept.promise,
    readPrivatePipe: pipe => {
      const item = pipe as typeof instances[number]
      const input = item.input
      item.reads.push(input)
      item.input = pending<Uint8Array>()
      void item.input.promise.catch(() => {})
      return input.promise
    },
    writePrivatePipe: async (pipe, bytes) => { (pipe as typeof instances[number]).output.push(Buffer.from(bytes)); return bytes.length },
    closePrivatePipe: async pipe => {
      const item = pipe as typeof instances[number]
      item.closed = true; for (const read of item.reads) read.reject({ win32Code: 995 }); item.accept.reject({ win32Code: 995 }); item.input.reject({ win32Code: 995 })
    },
  }
  return { instances, layer: windowsPrivatePipesLayer(native) }
}
const name = WindowsPipeName.make("\\\\.\\pipe\\magnitude-control-fixture")
const snapshot = Schema.decodeUnknownSync(ApplicationSnapshot)({ version: 1, pid: process.pid, endpoint: "http://127.0.0.1:11101", service: { _tag: "Starting", attempt: 0 }, tray: { _tag: "Registered" } })
const waitFor = (predicate: () => boolean) => Effect.repeat(Effect.sync(predicate), { until: Boolean, schedule: Schedule.spaced("1 millis") }).pipe(Effect.timeout("2 seconds"))

describe("Windows application control (simulated native transport)", () => {
  it("reports first-instance collision without waiting indefinitely for readiness", async () => {
    const layer = windowsPrivatePipesLayer({
      createPrivatePipe: () => { throw { win32Code: 5 } },
      acceptPrivatePipe: async () => { throw new Error("No pipe") },
      readPrivatePipe: async () => { throw new Error("No pipe") },
      writePrivatePipe: async () => { throw new Error("No pipe") },
      closePrivatePipe: async () => { throw new Error("No handle was acquired") },
    })
    const result = await Effect.runPromise(Effect.scoped(Effect.either(serveWindowsApplicationControl(name, {
      snapshot: Effect.succeed(snapshot), login: () => Effect.succeed({ _tag: "Disabled" }), dispatch: () => Effect.void,
    }))).pipe(Effect.provide(layer), Effect.timeout("2 seconds")))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("WindowsPipeFailed")
  })
  it.each(["EnsureRunning", "ShowWindow", "Observe", "Retry", "Quit"] as const)("retains a listener and writes the snapshot before dispatching %s", async intent => {
    const test = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const done = yield* Deferred.make<void>()
      yield* serveWindowsApplicationControl(name, {
        snapshot: Effect.succeed(snapshot), login: () => Effect.succeed({ _tag: "Disabled" }),
        dispatch: received => Effect.sync(() => {
          expect(received).toBe(intent)
          expect(JSON.parse(Buffer.concat(test.instances[0]!.output).toString()).tray._tag).toBe("Registered")
          expect(test.instances[1]!.closed).toBe(false)
        }).pipe(Effect.zipRight(Deferred.succeed(done, undefined))),
      })
      test.instances[0]!.input.resolve(Buffer.from(JSON.stringify({ version: 1, intent }) + "\n"))
      test.instances[0]!.accept.resolve(42)
      yield* Deferred.await(done).pipe(Effect.timeout("2 seconds"))
    })).pipe(Effect.provide(test.layer)))
    expect(test.instances.map(instance => instance.first)).toEqual([true, false])
    expect(test.instances.every(instance => instance.closed)).toBe(true)
  })

  it.each(["read", "enable", "disable"] as const)("uses the shared login contract for %s without dispatching intent", async login => {
    const test = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* serveWindowsApplicationControl(name, {
        snapshot: Effect.succeed(snapshot), dispatch: () => Effect.die("Login must not dispatch intent"),
        login: action => action === "disable" ? Effect.fail(new LoginStartupFailed({ message: "OS refused change" })) : Effect.succeed({ _tag: action === "read" ? "Disabled" : "Enabled" }),
      })
      test.instances[0]!.input.resolve(Buffer.from(JSON.stringify({ version: 1, login }) + "\n"))
      test.instances[0]!.accept.resolve(42)
      yield* waitFor(() => test.instances[0]!.closed)
      const response = JSON.parse(Buffer.concat(test.instances[0]!.output).toString())
      expect(response).toEqual(login === "disable" ? { _tag: "LoginStartupFailed", message: "OS refused change" } : { _tag: "LoginStartup", state: { _tag: login === "read" ? "Disabled" : "Enabled" } })
    })).pipe(Effect.provide(test.layer)))
  })

  it("rejects malformed framing without dispatching or losing the listener", async () => {
    const test = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* serveWindowsApplicationControl(name, { snapshot: Effect.succeed(snapshot), login: () => Effect.die("Invalid request"), dispatch: () => Effect.die("Invalid request") })
      test.instances[0]!.input.resolve(Buffer.from('{"version":1,"intent":"ReplaceOwner"}\n'))
      test.instances[0]!.accept.resolve(42)
      yield* waitFor(() => test.instances[0]!.closed)
      expect(test.instances[0]!.output).toEqual([])
      expect(test.instances[1]!.closed).toBe(false)
    })).pipe(Effect.provide(test.layer)))
  })

  it("bounds admission to sixteen clients and retires idle channels on interruption", async () => {
    const test = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const worker = yield* serveWindowsApplicationControl(name, { snapshot: Effect.succeed(snapshot), login: () => Effect.succeed({ _tag: "Disabled" }), dispatch: () => Effect.void })
      for (let index = 0; index < 16; index++) {
        yield* waitFor(() => test.instances.length > index)
        test.instances[index]!.accept.resolve(42)
      }
      yield* waitFor(() => test.instances.length === 17)
      test.instances[16]!.accept.resolve(42)
      yield* Effect.yieldNow()
      expect(test.instances.length).toBe(17)
      yield* waitFor(() => test.instances[0]!.reads.length > 0)
      test.instances[0]!.reads[0]!.resolve(Buffer.from('{"version":1,"intent":"Observe"}\n'))
      yield* waitFor(() => test.instances.length === 18)
      expect(test.instances.filter(instance => !instance.closed).length).toBe(17)
      yield* Fiber.interrupt(worker)
      expect(test.instances.every(instance => instance.closed)).toBe(true)
    })).pipe(Effect.provide(test.layer), Effect.timeout("3 seconds")))
  })
})
