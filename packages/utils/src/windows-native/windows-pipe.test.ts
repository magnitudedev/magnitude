import { Effect, Fiber, Option } from "effect"
import { describe, expect, it } from "vitest"
import { WindowsPipeName, WindowsPrivatePipes, windowsPrivatePipesLayer, type WindowsPipeBindings } from "./windows-pipe"

const pending = <A>() => {
  let resolve!: (value: A) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<A>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
const fixture = (overrides: Partial<WindowsPipeBindings> = {}) => {
  const observations = { creates: 0, closes: 0, reads: 0, writes: [] as Buffer[] }
  const native: WindowsPipeBindings = {
    createPrivatePipe: () => { observations.creates++; return {} },
    acceptPrivatePipe: async () => 42,
    readPrivatePipe: async () => { observations.reads++; return Buffer.from("read") },
    writePrivatePipe: async (_pipe, bytes) => { observations.writes.push(Buffer.from(bytes)); return bytes.length },
    closePrivatePipe: async () => { observations.closes++ },
    ...overrides,
  }
  return { native, observations, layer: windowsPrivatePipesLayer(native) }
}
const bind = Effect.flatMap(WindowsPrivatePipes, service => service.bind(WindowsPipeName.make("\\\\.\\pipe\\magnitude-fixture"), true))

describe("Windows pipe Effect ownership (simulated native boundary)", () => {
  it("chunks control writes, retains exact bytes and closes once", async () => {
    const test = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const pipe = yield* bind
      expect(yield* pipe.accept).toBe(42)
      expect(yield* pipe.read).toEqual(Buffer.from("read"))
      const bytes = Buffer.alloc(65537, 0xa7)
      yield* pipe.write(bytes)
      expect(test.observations.writes.map(bytes => bytes.length)).toEqual([65536, 1])
      expect(Buffer.concat(test.observations.writes)).toEqual(bytes)
      yield* Effect.all([pipe.close, pipe.close], { concurrency: "unbounded" })
      expect((yield* Effect.either(pipe.read))._tag).toBe("Left")
      expect(test.observations.reads).toBe(1)
    })).pipe(Effect.provide(test.layer)))
    expect(test.observations.closes).toBe(1)
  })

  it("serializes whole writes across partial native completions", async () => {
    const chunks: Buffer[] = []
    const test = fixture({ writePrivatePipe: async (_pipe, bytes) => {
      const count = Math.min(32768, bytes.length)
      chunks.push(Buffer.from(bytes.subarray(0, count)))
      await Promise.resolve()
      return count
    } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const pipe = yield* bind; yield* pipe.accept
      yield* Effect.all([pipe.write(Buffer.alloc(100000, 1)), pipe.write(Buffer.alloc(100000, 2))], { concurrency: "unbounded" })
    })).pipe(Effect.provide(test.layer)))
    const bytes = Buffer.concat(chunks)
    expect(bytes.length).toBe(200000)
    expect([...bytes.subarray(0, 100000)].every(value => value === bytes[0])).toBe(true)
    expect([...bytes.subarray(100000)].every(value => value !== bytes[0])).toBe(true)
  })

  it("interruption closes the native pipe and retires the pending read", async () => {
    const read = pending<Uint8Array>(); const started = pending<void>()
    let closes = 0
    const test = fixture({ readPrivatePipe: () => { started.resolve(); return read.promise }, closePrivatePipe: async () => { closes++; read.reject({ win32Code: 995 }) } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const pipe = yield* bind; yield* pipe.accept
      const fiber = yield* Effect.forkScoped(pipe.read)
      yield* Effect.promise(() => started.promise)
      yield* Fiber.interrupt(fiber)
      expect((yield* Effect.either(pipe.write(Buffer.from("late"))))._tag).toBe("Left")
    })).pipe(Effect.provide(test.layer)))
    expect(closes).toBe(1)
    expect(test.observations.writes).toEqual([])
  })

  it("cannot publish a late acceptance after explicit close", async () => {
    const accepted = pending<number>(); const started = pending<void>()
    const test = fixture({ acceptPrivatePipe: () => { started.resolve(); return accepted.promise } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const pipe = yield* bind
      const fiber = yield* Effect.forkScoped(Effect.either(pipe.accept))
      yield* Effect.promise(() => started.promise)
      yield* pipe.close
      accepted.resolve(42)
      const outcome = yield* Fiber.join(fiber)
      expect(outcome._tag).toBe("Left")
      expect((yield* Effect.either(pipe.read))._tag).toBe("Left")
    })).pipe(Effect.provide(test.layer)))
    expect(test.observations.closes).toBe(1)
    expect(test.observations.reads).toBe(0)
  })

  it("rejects duplicate acceptance without disrupting the admitted accept", async () => {
    const accepted = pending<number>(); const started = pending<void>()
    const test = fixture({ acceptPrivatePipe: () => { started.resolve(); return accepted.promise } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const pipe = yield* bind; const first = yield* Effect.forkScoped(pipe.accept)
      yield* Effect.promise(() => started.promise)
      expect((yield* Effect.either(pipe.accept))._tag).toBe("Left")
      expect(test.observations.closes).toBe(0)
      accepted.resolve(42)
      expect(yield* Fiber.join(first)).toBe(42)
    })).pipe(Effect.provide(test.layer)))
    expect(test.observations.closes).toBe(1)
  })

  it.each([0, -1, 65537, NaN])("rejects invalid native write count %s and closes", async count => {
    const test = fixture({ writePrivatePipe: async () => count })
    const outcome = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const pipe = yield* bind; yield* pipe.accept
      return yield* Effect.either(pipe.write(Buffer.from("message")))
    })).pipe(Effect.provide(test.layer)))
    expect(outcome._tag).toBe("Left")
    expect(test.observations.closes).toBe(1)
  })

  it("preserves native failure identity and acquires no handle on bind failure", async () => {
    const test = fixture({ createPrivatePipe: () => { throw { win32Code: 5 } } })
    const outcome = await Effect.runPromise(Effect.scoped(Effect.either(bind)).pipe(Effect.provide(test.layer)))
    expect(outcome._tag === "Left" && Option.getOrUndefined(outcome.left.win32Code)).toBe(5)
    expect(test.observations.closes).toBe(0)
  })
})
