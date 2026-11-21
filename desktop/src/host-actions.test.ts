import { describe, expect, it } from "vitest"
import { Chunk, Effect, Fiber, Stream } from "effect"
import { makeHostActions } from "./host-actions"

const first = <A, E>(stream: Stream.Stream<A, E>, count: number) => Stream.runCollect(Stream.take(stream, count)).pipe(Effect.map(Chunk.toArray))
const nothingWithin = <A, E>(stream: Stream.Stream<A, E>) => Effect.scoped(Effect.gen(function* () {
  const fiber = yield* Effect.fork(Stream.runCollect(Stream.take(stream, 1)))
  yield* Effect.sleep("20 millis")
  const done = yield* Fiber.poll(fiber)
  yield* Fiber.interrupt(fiber)
  return done._tag === "None"
}))

describe("host actions", () => {
  it("asks a renderer for no page until one is asked for, so the renderer decides where it opens", () => Effect.runPromise(Effect.gen(function* () {
    const actions = yield* makeHostActions
    expect(yield* nothingWithin(actions.subscribe)).toBe(true)
  })))

  it("gives a page asked for before the renderer subscribed to the next renderer, once", () => Effect.runPromise(Effect.gen(function* () {
    const actions = yield* makeHostActions
    yield* actions.navigate("usage")
    expect(yield* first(actions.subscribe, 1)).toEqual([{ _tag: "Navigate", page: "usage" }])
    expect(yield* nothingWithin(actions.subscribe)).toBe(true)
  })))

  it("forgets a page once a listening renderer received it, so a reload doesn't replay it", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const actions = yield* makeHostActions
    const listening = yield* Effect.fork(first(actions.subscribe, 1))
    yield* Effect.sleep("10 millis")
    yield* actions.navigate("connections")
    expect(yield* Fiber.join(listening)).toEqual([{ _tag: "Navigate", page: "connections" }])
    expect(yield* nothingWithin(actions.subscribe)).toBe(true)
  }))))

})
