import { Effect, Fiber, Stream, TestClock, TestContext } from "effect"
import { expect, it } from "vitest"
import { ApplicationMemory, applicationMemoryLayerFromLoader, observeApplicationMemory } from "./application-memory"

it("validates native measurements and recovers after observation failure", async () => {
  let attempt = 0
  const layer = applicationMemoryLayerFromLoader(() => ({ applicationMemory: async () => {
    if (attempt++ === 0) throw new Error("Process exited")
    return { bytes: 123456, processCount: 2, metric: "PhysicalFootprint" }
  } }))
  await Effect.runPromise(Effect.gen(function* () {
    const memory = yield* ApplicationMemory
    expect((yield* memory.read)._tag).toBe("Unavailable")
    const recovered = yield* memory.read
    expect(recovered).toMatchObject({ _tag: "Measured", bytes: 123456, processCount: 2 })
    if (recovered._tag === "Measured") expect(recovered.measuredAt).toBeGreaterThan(0)
  }).pipe(Effect.provide(layer)))
})
it.each([
  { bytes: -1, processCount: 1, metric: "PhysicalFootprint" },
  { bytes: Number.MAX_SAFE_INTEGER + 1, processCount: 1, metric: "PhysicalFootprint" },
  { bytes: 0, processCount: 0, metric: "PhysicalFootprint" },
  { bytes: 1, processCount: 1, metric: "Estimated" },
  { bytes: "100", processCount: 1, metric: "PhysicalFootprint" },
])("rejects invalid native telemetry instead of showing a fabricated total %#", async sample => {
  const state = await Effect.runPromise(Effect.flatMap(ApplicationMemory, memory => memory.read).pipe(
    Effect.provide(applicationMemoryLayerFromLoader(() => ({ applicationMemory: async () => sample }))),
  ))
  expect(state._tag).toBe("Unavailable")
})
it("missing native support remains a local observation failure", async () => {
  const result = await Effect.runPromise(Effect.flatMap(ApplicationMemory, memory => memory.read).pipe(
    Effect.provide(applicationMemoryLayerFromLoader(() => ({}))),
  ))
  expect(result._tag).toBe("Unavailable")
})
it("serializes simultaneous readers through the native observer", async () => {
  let active = 0, maximum = 0
  const layer = applicationMemoryLayerFromLoader(() => ({ applicationMemory: async () => {
    maximum = Math.max(maximum, ++active)
    await new Promise(resolve => setImmediate(resolve))
    active--
    return { bytes: 1, processCount: 1, metric: "PhysicalFootprint" }
  } }))
  const results = await Effect.runPromise(Effect.gen(function* () {
    const memory = yield* ApplicationMemory
    return yield* Effect.all([memory.read, memory.read, memory.read], { concurrency: "unbounded" })
  }).pipe(Effect.provide(layer)))
  expect(maximum).toBe(1)
  expect(results.every(result => result._tag === "Measured")).toBe(true)
})
it("samples only visible subscriptions and stops its cadence on cancellation", async () => {
  let visible = false, reads = 0
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const memory = ApplicationMemory.of({ read: Effect.sync(() => {
      reads++
      return { _tag: "Measured", bytes: 1, processCount: 1, measuredAt: 0, metric: "PhysicalFootprint" } as const
    }) })
    const fiber = yield* observeApplicationMemory(memory, () => visible).pipe(Stream.runDrain, Effect.forkScoped)
    yield* TestClock.adjust("9 seconds")
    expect(reads).toBe(0)
    visible = true
    yield* TestClock.adjust("3 seconds")
    expect(reads).toBe(1)
    visible = false
    yield* TestClock.adjust("6 seconds")
    expect(reads).toBe(1)
    yield* Fiber.interrupt(fiber)
    visible = true
    yield* TestClock.adjust("9 seconds")
    expect(reads).toBe(1)
  })).pipe(Effect.provide(TestContext.TestContext)))
})
