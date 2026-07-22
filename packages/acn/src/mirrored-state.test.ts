import { describe, expect, it } from "vitest"
import { Effect, Fiber, Option, Ref, Schema, Stream } from "effect"
import { defineMirroredState } from "@magnitudedev/protocol"
import { makeIcnObservedState } from "@magnitudedev/icn"
import { bindMirroredState, MirroredStateChanges, MirroredStateChangesLive, makeMirroredState } from "./mirrored-state"

const CountMirror = defineMirroredState("GetTestCount", {
  stateSchema: Schema.Struct({ count: Schema.Number }),
  errorSchema: Schema.Never,
})

const OperationIdsMirror = defineMirroredState("GetTestOperationIds", {
  stateSchema: Schema.Struct({ operationIds: Schema.Array(Schema.String) }),
  errorSchema: Schema.Never,
})

describe("mirrored state", () => {
  it("updates the snapshot and publishes the matching revision", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const changes = yield* MirroredStateChanges
      const mirror = yield* makeMirroredState(CountMirror, { count: 0 })
      const event = yield* Effect.fork(Stream.runHead(changes.stream))
      yield* Effect.sleep("1 millis")
      const snapshot = yield* mirror.update((state) => ({ count: state.count + 1 }))
      const invalidation = yield* event
      return { snapshot, invalidation }
    }).pipe(Effect.provide(MirroredStateChangesLive))))

    expect(result.snapshot).toEqual({ revision: 1, state: { count: 1 } })
    expect(Option.getOrThrow(result.invalidation)).toEqual({ _tag: "changed", id: "GetTestCount", revision: 1 })
  })

  it("atomically returns a result from a transition", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const initial: { operationIds: string[] } = { operationIds: [] }
      const mirror = yield* makeMirroredState(OperationIdsMirror, initial)
      return yield* mirror.modify((state) => ({
        state: { operationIds: [...state.operationIds, "op-1"] },
        result: "op-1",
      }))
    }).pipe(Effect.provide(MirroredStateChangesLive)))

    expect(result).toEqual({
      snapshot: { revision: 1, state: { operationIds: ["op-1"] } },
      result: "op-1",
    })
  })

  it("does not publish or store a transition that defects", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const mirror = yield* makeMirroredState(CountMirror, { count: 0 })
      const exit = yield* Effect.exit(mirror.update(() => {
        throw new Error("invalid transition")
      }))
      const snapshot = yield* mirror.get
      return { exit, snapshot }
    }).pipe(Effect.provide(MirroredStateChangesLive)))

    expect(result.exit._tag).toBe("Failure")
    expect(result.snapshot).toEqual({ revision: 0, state: { count: 0 } })
  })

  it("suppresses a no-op transition", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const changes = yield* MirroredStateChanges
      const mirror = yield* makeMirroredState(CountMirror, { count: 0 })
      const event = yield* Effect.fork(Stream.runHead(changes.stream))
      yield* Effect.yieldNow()
      const snapshot = yield* mirror.setIfChanged({ count: 0 }, (left, right) => left.count === right.count)
      const published = yield* Fiber.poll(event)
      yield* Fiber.interrupt(event)
      return { snapshot, published }
    }).pipe(Effect.provide(MirroredStateChangesLive))))

    expect(result.snapshot).toEqual({ revision: 0, state: { count: 0 } })
    expect(Option.isNone(result.published)).toBe(true)
  })

  it("binds an authoritative source without copying or re-versioning it", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const changes = yield* MirroredStateChanges
      const sourceValue = yield* Ref.make({ count: 0 })
      const source = yield* makeIcnObservedState(
        { count: 0 },
        Ref.get(sourceValue),
        (left, right) => left.count === right.count,
      )
      const events = yield* changes.stream.pipe(Stream.take(1), Stream.runCollect, Effect.fork)
      yield* Effect.yieldNow()
      const mirror = yield* bindMirroredState(CountMirror, source)
      yield* Effect.yieldNow()

      yield* Ref.set(sourceValue, { count: 1 })
      yield* source.refresh

      return {
        events: Array.from(yield* Fiber.join(events)),
        snapshot: yield* mirror.get,
      }
    }).pipe(Effect.provide(MirroredStateChangesLive))))

    expect(result.events).toEqual([
      { _tag: "changed", id: "GetTestCount", revision: 1 },
    ])
    expect(result.snapshot).toEqual({ revision: 1, state: { count: 1 } })
  })
})
