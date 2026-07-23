import { describe, expect, it } from "vitest"
import { Effect, Fiber, Option, Ref, Stream } from "effect"
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
      const observed = yield* makeIcnObservedState(
        { value: 1 },
        Effect.succeed({ value: 1 }),
        (left, right) => left.value === right.value,
      )
      const snapshots = yield* observed.changes.pipe(Stream.take(2), Stream.runCollect, Effect.fork)
      yield* Effect.yieldNow()

      yield* observed.refresh
      const initialized = yield* observed.initialized
      const first = yield* Fiber.join(snapshots)
      const next = yield* Effect.fork(observed.changes.pipe(Stream.drop(2), Stream.runHead))
      yield* observed.refresh
      yield* Effect.yieldNow()
      const polled = yield* Fiber.poll(next)
      yield* Fiber.interrupt(next)
      return { first, initialized, polled, snapshot: yield* observed.get }
    })))

    expect(Array.from(result.first)).toEqual([
      { revision: 0, state: { value: 1 } },
      { revision: 1, state: { value: 1 } },
    ])
    expect(result.initialized).toBe(true)
    expect(Option.isNone(result.polled)).toBe(true)
    expect(result.snapshot).toEqual({ revision: 1, state: { value: 1 } })
  })
})
