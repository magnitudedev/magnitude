import { Cause, Clock, DateTime, Deferred, Effect, Either, Fiber, Option, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { EventStore } from "@magnitudedev/event-engine"
import { type ResponseEnded, responseMetrics, type SessionEvent } from "@magnitudedev/acn-protocol"
import { done, newId, scriptedInference, sent, started, storeFailure, text, thought } from "./chat.test-support"
import { isResponse, Transcript } from "./folds"
import { Generation } from "./generation"
import { ChatEngineDefinition } from "./service"

// Generation at the engine, where tasks can be queued behind a running one.

const endings = (events: ReadonlyArray<SessionEvent>) =>
  events.flatMap((event) => event._tag === "response_ended" ? [[event.responseId, event.outcome] as const] : [])

/** An open chat engine whose responses stream one delta, then wait until they are interrupted. */
const blocked = Effect.gen(function* () {
  const streaming = yield* Deferred.make<void>()
  const inference = yield* scriptedInference(() =>
    Stream.concat(Stream.make(text("a")), Stream.fromEffect(Deferred.succeed(streaming, undefined)).pipe(Stream.drain, Stream.concat(Stream.never))))
  const store = yield* EventStore.memory<SessionEvent>()
  const runtime = yield* ChatEngineDefinition.open({ store }).pipe(Effect.provide(inference.layer))
  const [first, second] = [newId(), newId()]
  // Two responses at once, which commands never allow, so the second waits in the lane.
  yield* runtime.transact(() => Effect.succeed({
    events: [sent(newId(), "Hi", 0), started(first, 0), started(second, 0)],
    result: undefined,
  }))
  yield* Deferred.await(streaming)
  return { runtime, store, inference, first, second }
})

const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))

describe("Generation", () => {
  it("ends a stopped running response with what it streamed, and a queued one without streaming", () => run(Effect.gen(function* () {
    const { runtime, store, inference, first, second } = yield* blocked
    yield* runtime.transact(() => Effect.succeed({ events: [], interrupts: [Generation], result: undefined }))
    yield* runtime.awaitIdle
    const events = yield* store.events
    expect(endings(events)).toEqual([[first, { _tag: "stopped", by: "user" }], [second, { _tag: "stopped", by: "user" }]])
    expect(events.filter((event) => event._tag === "response_output")).toHaveLength(1)
    expect(inference.requests).toHaveLength(1)
    // Only the response that reached inference has serving usage.
    expect(inference.usage.map((record) => record.id)).toEqual([`chat:${first}`])
  })))

  it("ends running and queued responses as shut down when the engine closes", () => run(Effect.gen(function* () {
    const { runtime, store, inference, first, second } = yield* blocked
    yield* runtime.close("closed")
    expect(endings(yield* store.events)).toEqual([
      [first, { _tag: "stopped", by: "shutdown" }],
      [second, { _tag: "stopped", by: "shutdown" }],
    ])
    expect((yield* runtime.closed)._tag).toBe("Closed")
    expect(inference.usage.map((record) => record.id)).toEqual([`chat:${first}`])
  })))

  it("records nothing for a removed chat, but the serving usage of the response that reached inference", () => run(Effect.gen(function* () {
    const { runtime, store, inference, first } = yield* blocked
    yield* runtime.close("removed")
    expect(endings(yield* store.events)).toEqual([])
    expect(inference.usage).toEqual([expect.objectContaining({ id: `chat:${first}`, complete: false })])
    expect(Either.isLeft(yield* Effect.either(runtime.accept(sent(newId(), "late", 1))))).toBe(true)
  })))

  /** An engine whose store refuses every commit of these events, streaming these chunks. */
  const refusing = (refused: SessionEvent["_tag"], chunks: Parameters<typeof Stream.make>) => Effect.gen(function* () {
    const inference = yield* scriptedInference(() => Stream.make(...chunks))
    const memory = yield* EventStore.memory<SessionEvent>()
    const store = {
      ...memory,
      commit: (events: ReadonlyArray<SessionEvent>) =>
        events.some((event) => event._tag === refused) ? Effect.fail(storeFailure) : memory.commit(events),
    }
    const runtime = yield* ChatEngineDefinition.open({ store }).pipe(Effect.provide(inference.layer))
    yield* runtime.transact(() => Effect.succeed({ events: [sent(newId(), "Hi", 0), started(newId(), 0)], result: undefined }))
    return { runtime, events: memory.events, usage: inference.usage }
  })

  it("ends a response whose output it could not record, while streaming or at its end, as failed for storage", () => run(Effect.gen(function* () {
    for (const chunks of [[thought("Hm"), text("Hi"), done()], [text("Hi"), done()]] as const) {
      const { runtime, events } = yield* refusing("response_output", [...chunks])
      yield* runtime.awaitIdle
      const [end] = (yield* events).filter((event): event is ResponseEnded => event._tag === "response_ended")
      expect(end?.outcome).toMatchObject({ _tag: "failed", error: { _tag: "storage", message: "Chat history could not be read." } })
    }
  })))

  it("fails the engine when it cannot record a response's end, leaving the end to recovery", () => run(Effect.gen(function* () {
    const { runtime, events, usage } = yield* refusing("response_ended", [text("Hi"), done()])
    expect((yield* runtime.closed)._tag).toBe("Failed")
    expect(endings(yield* events)).toEqual([])
    // Its outcome was decided, so its serving usage is recorded all the same.
    expect(usage).toEqual([expect.objectContaining({ complete: true, input: 40, output: 3 })])
  })))

  it("fails the engine with a defect raised while a stop is pending, rather than ending the response as stopped", () => run(Effect.gen(function* () {
    const committing = yield* Deferred.make<void>()
    const proceed = yield* Deferred.make<void>()
    const stopped = yield* Deferred.make<void>()
    // The stream's end, once interrupted, waits for the stop to have been delivered.
    const inference = yield* scriptedInference(() =>
      Stream.concat(Stream.make(text("a"), thought("b")), Stream.never).pipe(Stream.ensuring(Deferred.await(stopped))))
    const memory = yield* EventStore.memory<SessionEvent>()
    const defects = yield* Ref.make(1)
    // The first output's commit waits, then dies; a later one is recorded.
    const store = {
      ...memory,
      commit: (events: ReadonlyArray<SessionEvent>) => events.some((event) => event._tag === "response_output")
        ? Effect.flatMap(Ref.getAndSet(defects, 0), (left) => left > 0
            ? Effect.zipRight(Deferred.succeed(committing, undefined), Effect.zipRight(Deferred.await(proceed), Effect.dieMessage("commit defect")))
            : memory.commit(events))
        : memory.commit(events),
    }
    const runtime = yield* ChatEngineDefinition.open({ store }).pipe(Effect.provide(inference.layer))
    yield* runtime.transact(() => Effect.succeed({ events: [sent(newId(), "Hi", 0), started(newId(), 0)], result: undefined }))
    yield* Deferred.await(committing)
    // The stop waits for the commit's lock, then is delivered while the defect unwinds the stream.
    const stop = yield* Effect.fork(runtime.transact(() => Effect.succeed({ events: [], interrupts: [Generation], result: undefined })))
    yield* Deferred.succeed(proceed, undefined)
    yield* Fiber.join(stop)
    yield* Deferred.succeed(stopped, undefined)
    yield* runtime.awaitIdle
    expect(endings(yield* memory.events)).toEqual([])
    const closure = yield* runtime.closed
    expect(closure._tag === "Failed" && Cause.pretty(closure.cause)).toContain("commit defect")
  })))

  it("dates every time of a response from one anchor, so a wall clock stepped back reorders none of them", () => {
    // A wall clock that steps back a second at every reading, from before the response's recorded
    // start; the monotonic clock is the real one.
    const real = Clock.make()
    let wall = 1_000_000
    const stepping: Clock.Clock = {
      [Clock.ClockTypeId]: Clock.ClockTypeId,
      unsafeCurrentTimeMillis: () => (wall -= 1_000),
      currentTimeMillis: Effect.sync(() => (wall -= 1_000)),
      unsafeCurrentTimeNanos: () => real.unsafeCurrentTimeNanos(),
      currentTimeNanos: real.currentTimeNanos,
      sleep: (duration) => real.sleep(duration),
    }
    const startedAt = 2_000_000
    return run(Effect.withClock(stepping)(Effect.gen(function* () {
      const inference = yield* scriptedInference(() => Stream.make(text("a"), text("b"), text("c"), thought("d"), done()))
      const store = yield* EventStore.memory<SessionEvent>()
      const runtime = yield* ChatEngineDefinition.open({ store }).pipe(Effect.provide(inference.layer))
      const response = newId()
      yield* runtime.transact(() => Effect.succeed({ events: [sent(newId(), "Hi", startedAt), started(response, startedAt)], result: undefined }))
      yield* runtime.awaitIdle
      const events = yield* store.events
      const offsets = events.flatMap((event) => event._tag === "response_output" ? [event.at] : [])
      expect(offsets).toHaveLength(2)
      for (const at of offsets) {
        expect(at[0]).toBe(0)
        expect(at.every((offset, index) => offset >= 0 && offset >= (at[index - 1] ?? 0))).toBe(true)
      }
      expect(endings(events)).toEqual([[response, { _tag: "finished", finishReason: "stop" }]])
      // Every recorded time is at or after the response's start, in the order it was observed.
      const times = events.flatMap((event) => event._tag === "response_output" || event._tag === "response_ended" ? [DateTime.toEpochMillis(event.time)] : [])
      expect(times[0]).toBeGreaterThanOrEqual(startedAt)
      expect(times.every((time, index) => time >= (times[index - 1] ?? startedAt))).toBe(true)
      const message = (yield* runtime.read.get(Transcript)).find(isResponse(response))!
      const metrics = responseMetrics(message)
      for (const duration of [metrics.ttftMs, metrics.firstAnswerMs, metrics.thoughtForMs, metrics.totalMs]) {
        expect(Option.getOrThrow(duration)).toBeGreaterThanOrEqual(0)
      }
      expect(DateTime.toEpochMillis(Option.getOrThrow(message.end).time)).toBe(times.at(-1))
      expect(inference.usage).toEqual([expect.objectContaining({ complete: true, completedAt: times.at(-1), firstTokenMs: expect.any(Number) })])
      expect(inference.usage[0]!.firstTokenMs).toBeGreaterThanOrEqual(0)
    })))
  })
})
