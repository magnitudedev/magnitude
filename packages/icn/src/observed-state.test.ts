import { describe, expect, it } from "vitest"
import { Effect, Fiber, Option, Ref, Stream } from "effect"
import { makeIcnObservedState } from "./observed-state"

describe("ICN observed state", () => {
  it("replays the current snapshot and revisions only structurally changed reads", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const source = yield* Ref.make({ value: 1 })
      const observed = yield* makeIcnObservedState({ value: 1 }, Ref.get(source))
      const snapshots = yield* observed.changes.pipe(Stream.take(2), Stream.runCollect, Effect.fork)
      yield* Effect.yieldNow()

      yield* observed.refresh
      yield* Ref.set(source, { value: 2 })
      yield* observed.refresh

      return yield* Fiber.join(snapshots)
    })))

    expect(Array.from(result)).toEqual([
      { revision: 0, state: { value: 1 } },
      { revision: 1, state: { value: 2 } },
    ])
  })

  it("does not emit an equal refresh", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const observed = yield* makeIcnObservedState({ value: 1 }, Effect.succeed({ value: 1 }))
      const initial = yield* Stream.runHead(observed.changes)
      const next = yield* Effect.fork(observed.changes.pipe(Stream.drop(1), Stream.runHead))

      yield* observed.refresh
      yield* Effect.yieldNow()
      const polled = yield* Fiber.poll(next)
      yield* Fiber.interrupt(next)
      return { initial, polled, snapshot: yield* observed.get }
    })))

    expect(Option.getOrThrow(result.initial)).toEqual({ revision: 0, state: { value: 1 } })
    expect(Option.isNone(result.polled)).toBe(true)
    expect(result.snapshot).toEqual({ revision: 0, state: { value: 1 } })
  })
})
