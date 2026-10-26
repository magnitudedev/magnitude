import { Effect, HashMap, Option } from "effect"
import { describe, expect, it } from "vitest"
import { EventStore } from "@magnitudedev/event-engine"
import type { ChatContentPart, ChatMessage, SessionEvent } from "@magnitudedev/acn-protocol"
import { at, finished, newId, output, sent, started } from "./chat.test-support"
import { ChatEngine, ResponseActivity } from "./engine"
import { lastObserved, Live, Seen, Transcript } from "./folds"

const Folds = ChatEngine.make({ name: "folds", projections: [Seen, Transcript, Live], workers: [] })

/** The folds over these events, and the store they were committed to. */
const folded = (events: ReadonlyArray<SessionEvent>) => Effect.gen(function* () {
  const store = yield* EventStore.memory<SessionEvent>()
  const runtime = yield* Folds.open({ store })
  yield* Effect.forEach(events, runtime.accept, { discard: true })
  return { runtime, store }
})

const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))
const part = (tag: "text" | "reasoning", value: string, from: number, to: number): ChatContentPart =>
  ({ _tag: tag, text: value, time: { started: at(from), ended: at(to) } })

describe("Seen", () => {
  it("names what each ID was used for: a retry answers the message the response it replaced did", () => run(Effect.gen(function* () {
    const [message, response, retry] = [newId(), newId(), newId()]
    const { runtime } = yield* folded([
      sent(message, "Hello", 1_000),
      started(response, 1_000),
      output(response, "text", 1_100, ["Hi"], [0]),
      finished(response, 1_200),
      started(retry, 1_300, Option.some(response)),
    ])
    const seen = yield* runtime.read.get(Seen)
    expect(Object.fromEntries(HashMap.toEntries(seen.ids))).toEqual({
      [message]: { _tag: "message" },
      [response]: { _tag: "response", message },
      [retry]: { _tag: "response", message },
    })
    expect(seen.latest).toEqual(Option.some(message))
  })))
})

describe("Transcript", () => {
  it("joins consecutive output of one part, each delta at its own time, and records the end", () => run(Effect.gen(function* () {
    const [message, response] = [newId(), newId()]
    const { runtime } = yield* folded([
      sent(message, "Hi", 0),
      started(response, 10),
      output(response, "reasoning", 20, ["Let", " me"], [0, 30]),
      output(response, "reasoning", 60, [" think"], [0]),
      output(response, "text", 100, ["Hello"], [0]),
      finished(response, 200),
    ])
    const [user, assistant] = yield* runtime.read.get(Transcript)
    expect(user).toEqual({ _tag: "user", id: message, text: "Hi", time: at(0) })
    if (assistant?._tag !== "assistant") throw new Error("Expected the response")
    expect(assistant.content).toEqual([part("reasoning", "Let me think", 20, 60), part("text", "Hello", 100, 100)])
    expect(assistant.started).toEqual(at(10))
    expect(Option.map(assistant.end, (end) => end.outcome)).toEqual(Option.some({ _tag: "finished", finishReason: "stop" }))
  })))

  it("replaces the latest response with its retry, keeping every other message as it was", () => run(Effect.gen(function* () {
    const [message, response, retry] = [newId(), newId(), newId()]
    const { runtime } = yield* folded([sent(message, "Hi", 0), started(response, 10), finished(response, 20)])
    const [user] = yield* runtime.read.get(Transcript)
    yield* runtime.accept(started(retry, 30, Option.some(response)))
    const transcript = yield* runtime.read.get(Transcript)
    expect(transcript.map((m) => m.id)).toEqual([message, retry])
    expect(transcript[0]).toBe(user)
  })))
})

describe("Live", () => {
  it("follows the running response's activity, moves recorded output to the transcript, and clears at its end", () => run(Effect.gen(function* () {
    const response = newId()
    const { runtime } = yield* folded([sent(newId(), "Hi", 0), started(response, 10)])
    expect(yield* runtime.read.get(Live)).toEqual(Option.some({
      id: response,
      phase: Option.none(),
      usage: Option.none(),
      timings: Option.none(),
      pending: Option.none(),
    }))
    const phase = { phase: { _tag: "generating" as const }, progress: Option.none() }
    yield* runtime.update(ResponseActivity, Option.some({ _tag: "phase", responseId: response, phase }))
    yield* runtime.update(ResponseActivity, Option.some({ _tag: "output", responseId: response, part: part("text", "He", 20, 20) }))
    yield* runtime.update(ResponseActivity, Option.some({ _tag: "output", responseId: response, part: part("text", "llo", 25, 25) }))
    const live = Option.getOrThrow(yield* runtime.read.get(Live))
    expect(live.phase).toEqual(Option.some(phase))
    expect(live.pending).toEqual(Option.some(part("text", "Hello", 20, 25)))

    yield* runtime.accept(output(response, "text", 20, ["He", "llo"], [0, 5]))
    expect(Option.flatMap(yield* runtime.read.get(Live), (l) => l.pending)).toEqual(Option.none())
    const transcript = yield* runtime.read.get(Transcript)
    expect(transcript.at(-1)).toMatchObject({ content: [part("text", "Hello", 20, 25)] })

    yield* runtime.accept(finished(response, 40))
    expect(yield* runtime.read.get(Live)).toEqual(Option.none())
  })))

  it("ignores activity for a response that is not the running one", () => run(Effect.gen(function* () {
    const response = newId()
    const { runtime } = yield* folded([sent(newId(), "Hi", 0), started(response, 10)])
    const before = yield* runtime.read.get(Live)
    yield* runtime.update(ResponseActivity, Option.some({ _tag: "output", responseId: newId(), part: part("text", "x", 1, 1) }))
    expect(yield* runtime.read.get(Live)).toBe(before)
  })))
})

describe("lastObserved", () => {
  it("is the response's start, or the end of its last recorded output when that is later", () => run(Effect.gen(function* () {
    const response = newId()
    const latest = (transcript: ReadonlyArray<ChatMessage>) => {
      const last = transcript.at(-1)
      if (last?._tag !== "assistant") throw new Error("Expected the response")
      return lastObserved(last)
    }
    // A response with no recorded output was last observed at its start.
    const idle = yield* folded([sent(newId(), "Hi", 0), started(response, 10)])
    expect(latest(yield* idle.runtime.read.get(Transcript))).toEqual(at(10))
    const { runtime } = yield* folded([sent(newId(), "Hi", 0), started(response, 10), output(response, "text", 20, ["a", "b"], [0, 700])])
    expect(latest(yield* runtime.read.get(Transcript))).toEqual(at(720))
  })))
})
