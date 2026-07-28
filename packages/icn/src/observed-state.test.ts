import { describe, expect, it } from "vitest"
import { Effect, Fiber, Ref, Stream } from "effect"
import { makeIcnObservedState } from "./observed-state"

describe("ICN observed state", () => {
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
})
