import { describe, expect, it } from "vitest"
import { Effect, Option, Stream } from "effect"
import { makeMirroredResource } from "./mirrored-resource"

describe("mirrored resource", () => {
  it("updates the snapshot and publishes the matching revision", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const resource = yield* makeMirroredResource({ count: 0 })
      const event = yield* Effect.fork(Stream.runHead(resource.changes))
      yield* Effect.yieldNow()
      const snapshot = yield* resource.update((state) => ({ count: state.count + 1 }))
      const invalidation = yield* event
      return { snapshot, invalidation }
    })))

    expect(result.snapshot).toEqual({ revision: 1, state: { count: 1 } })
    expect(Option.getOrThrow(result.invalidation)).toEqual({ _tag: "changed", revision: 1 })
  })

  it("does not publish or store a transition that defects", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const resource = yield* makeMirroredResource({ count: 0 })
      const exit = yield* Effect.exit(resource.update(() => {
        throw new Error("invalid transition")
      }))
      const snapshot = yield* resource.get
      return { exit, snapshot }
    }))

    expect(result.exit._tag).toBe("Failure")
    expect(result.snapshot).toEqual({ revision: 0, state: { count: 0 } })
  })
})
