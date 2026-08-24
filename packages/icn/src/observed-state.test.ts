import { describe, expect, it } from "vitest"
import { Effect, Fiber, Ref, Stream } from "effect"
import { makeIcnObservedState } from "./observed-state"

describe("ICN observed state", () => {
  it("marks an authoritative initial read as initialized without rereading it", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const reads = yield* Ref.make(0)
      const observed = yield* makeIcnObservedState(
        { value: 1 },
        Ref.update(reads, (count) => count + 1).pipe(Effect.as({ value: 1 })),
        (left, right) => left.value === right.value,
        { initiallyInitialized: true },
      )
      const initial = yield* observed.get
      yield* observed.refresh
      return {
        initialized: yield* observed.initialized,
        initial,
        current: yield* observed.get,
        reads: yield* Ref.get(reads),
      }
    }))

    expect(result).toEqual({
      initialized: true,
      initial: { revision: 1, state: { value: 1 } },
      current: { revision: 1, state: { value: 1 } },
      reads: 1,
    })
  })

  it("publishes initial readiness before structurally changed reads", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const source = yield* Ref.make({ value: 1 })
      const observed = yield* makeIcnObservedState(
        { value: 1 },
        Ref.get(source),
        (left, right) => left.value === right.value,
      )
      const snapshots = yield* observed.changes.pipe(Stream.take(3), Stream.runCollect, Effect.fork)
      yield* Effect.yieldNow()

      yield* observed.refresh
      yield* Ref.set(source, { value: 2 })
      yield* observed.refresh

      return yield* Fiber.join(snapshots)
    })))

    expect(Array.from(result)).toEqual([
      { revision: 0, state: { value: 1 } },
      { revision: 1, state: { value: 1 } },
      { revision: 2, state: { value: 2 } },
    ])
  })

  it("emits the first equal refresh but suppresses later equal refreshes", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const source = yield* Ref.make({ value: 1 })
      const observed = yield* makeIcnObservedState(
        { value: 1 },
        Ref.get(source),
        (left, right) => left.value === right.value,
      )
      const snapshots = yield* observed.changes.pipe(Stream.take(3), Stream.runCollect, Effect.fork)
      yield* Effect.yieldNow()

      yield* observed.refresh
      yield* observed.refresh
      yield* Ref.set(source, { value: 2 })
      yield* observed.refresh
      return {
        snapshots: Array.from(yield* Fiber.join(snapshots)),
        initialized: yield* observed.initialized,
        snapshot: yield* observed.get,
      }
    })))

    expect(result.snapshots).toEqual([
      { revision: 0, state: { value: 1 } },
      { revision: 1, state: { value: 1 } },
      { revision: 2, state: { value: 2 } },
    ])
    expect(result.initialized).toBe(true)
    expect(result.snapshot).toEqual({ revision: 2, state: { value: 2 } })
  })

  it("publishes operation evidence immediately without reading", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const reads = yield* Ref.make(0)
      const observed = yield* makeIcnObservedState(
        { value: 1 },
        Ref.updateAndGet(reads, (count) => count + 1).pipe(
          Effect.as({ value: 99 }),
        ),
        (left, right) => left.value === right.value,
      )

      yield* observed.update(() => ({ value: 2 }))

      return {
        reads: yield* Ref.get(reads),
        snapshot: yield* observed.get,
      }
    }))

    expect(result).toEqual({
      reads: 0,
      snapshot: { revision: 1, state: { value: 2 } },
    })
  })
})
