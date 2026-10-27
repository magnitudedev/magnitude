import { DateTime, Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { generateMessageId } from "./identifier"
import {
  SessionEvent,
  StoredMessageSent,
  StoredResponseEnded,
  StoredResponseOutput,
  StoredResponseStarted,
} from "./session-event"

const messageId = Effect.runSync(generateMessageId)
const responseId = Effect.runSync(generateMessageId)
const replaced = Effect.runSync(generateMessageId)
const time = DateTime.unsafeMake(1_761_350_400_123)
const epoch = 1_761_350_400_123

const settings = {
  systemPrompt: "Be brief.",
  temperature: Option.none(),
  maxOutputTokens: Option.some(512),
  topP: Option.none(),
  topK: Option.none(),
  minP: Option.none(),
  repetitionPenalty: Option.none(),
  presencePenalty: Option.none(),
  frequencyPenalty: Option.none(),
  seed: Option.none(),
  stop: [],
}

// Each runtime event with the exact JSON it encodes to.
const cases: ReadonlyArray<readonly [SessionEvent, unknown]> = [
  [
    { _tag: "message_sent", time, messageId, text: "Hello" },
    { type: "message_sent", time: epoch, messageId, text: "Hello" },
  ],
  [
    {
      _tag: "response_started",
      time,
      responseId,
      replaces: Option.some(replaced),
      model: "qwen3-8b" as never,
      reasoningEffort: Option.none(),
      settings,
    },
    {
      type: "response_started",
      time: epoch,
      responseId,
      replaces: replaced,
      model: "qwen3-8b",
      settings: { systemPrompt: "Be brief.", maxOutputTokens: 512, stop: [] },
    },
  ],
  [
    {
      _tag: "response_output",
      time,
      responseId,
      part: "reasoning",
      deltas: ["Let", " me", " think"],
      at: [0, 21, 40],
    },
    {
      type: "response_output",
      time: epoch,
      responseId,
      part: "reasoning",
      deltas: ["Let", " me", " think"],
      at: [0, 21, 40],
    },
  ],
  [
    {
      _tag: "response_ended",
      time,
      responseId,
      outcome: { _tag: "finished", finishReason: "stop" },
      usage: Option.some({ promptTokens: 20, cachedTokens: 0, completionTokens: 3, draftTokens: Option.some(4), draftAcceptedTokens: Option.some(2) }),
      timings: Option.some({ promptMs: 31.5, predictedMs: 40, samplerMs: 0.4, parserMs: 0.1, timeToFirstTokenMs: 35 }),
    },
    {
      type: "response_ended",
      time: epoch,
      responseId,
      outcome: { type: "finished", finishReason: "stop" },
      usage: { promptTokens: 20, cachedTokens: 0, completionTokens: 3, draftTokens: 4, draftAcceptedTokens: 2 },
      timings: { promptMs: 31.5, predictedMs: 40, samplerMs: 0.4, parserMs: 0.1, timeToFirstTokenMs: 35 },
    },
  ],
  [
    {
      _tag: "response_ended",
      time,
      responseId,
      outcome: { _tag: "failed", error: { _tag: "inference", code: "server_error", message: "closed", param: Option.some("model") } },
      usage: Option.none(),
      timings: Option.none(),
    },
    {
      type: "response_ended",
      time: epoch,
      responseId,
      outcome: { type: "failed", error: { type: "inference", code: "server_error", message: "closed", param: "model" } },
    },
  ],
  [
    {
      _tag: "response_ended",
      time,
      responseId,
      outcome: { _tag: "failed", error: { _tag: "unreachable", message: "closed" } },
      usage: Option.none(),
      timings: Option.none(),
    },
    {
      type: "response_ended",
      time: epoch,
      responseId,
      outcome: { type: "failed", error: { type: "unreachable", message: "closed" } },
    },
  ],
  [
    {
      _tag: "response_ended",
      time,
      responseId,
      outcome: { _tag: "stopped", by: "shutdown" },
      usage: Option.none(),
      timings: Option.none(),
    },
    { type: "response_ended", time: epoch, responseId, outcome: { type: "stopped", by: "shutdown" } },
  ],
]

const hasKey = (value: unknown, key: string): boolean =>
  Array.isArray(value)
    ? value.some((item) => hasKey(item, key))
    : typeof value === "object" && value !== null
      && (key in value || Object.values(value).some((item) => hasKey(item, key)))

describe("SessionEvent", () => {
  it.each(cases)("round-trips $0._tag through its stored JSON", (event, stored) => {
    const encoded = Schema.encodeSync(SessionEvent)(event)
    const json = JSON.parse(JSON.stringify(encoded))
    expect(json).toStrictEqual(stored)
    expect(hasKey(json, "_tag")).toBe(false)
    expect(Schema.decodeUnknownSync(SessionEvent)(json)).toStrictEqual(event)
  })

  it("covers every event type", () => {
    expect(new Set(cases.map(([event]) => event._tag)).size).toBe(SessionEvent.members.length)
  })

  it("rejects output whose arrival times do not match its deltas", () => {
    const base = { type: "response_output", time: epoch, responseId, part: "text" }
    const decode = Schema.decodeUnknownEither(SessionEvent)
    expect(decode({ ...base, deltas: ["a", "b"], at: [0] })._tag).toBe("Left")
    expect(decode({ ...base, deltas: ["a"], at: [5] })._tag).toBe("Left")
    expect(decode({ ...base, deltas: [], at: [] })._tag).toBe("Left")
  })


  it.each(cases)("stores $0._tag at version 1 under its type's stored schema", (event, stored) => {
    const schema = storedSchemaOf(event._tag)
    const encoded = Schema.encodeSync(schema)(event)
    expect(JSON.parse(JSON.stringify(encoded))).toStrictEqual({ version: 1, value: stored })
    expect(Schema.decodeUnknownSync(schema)({ version: 1, value: stored })).toStrictEqual(event)
  })

  it("rejects a version its type does not list", () => {
    const [event, stored] = cases[0]!
    expect(Schema.decodeUnknownEither(storedSchemaOf(event._tag))({ version: 2, value: stored })._tag).toBe("Left")
  })
})

// The stored schema of an event's type, for any event.
const storedSchemaOf = (type: SessionEvent["_tag"]) =>
  storedSchemas[type] as unknown as Schema.Schema<SessionEvent, { readonly version: number; readonly value: unknown }>

const storedSchemas = {
  message_sent: StoredMessageSent,
  response_started: StoredResponseStarted,
  response_output: StoredResponseOutput,
  response_ended: StoredResponseEnded,
} as const
