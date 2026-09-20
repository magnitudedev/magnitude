import { expect, test } from "vitest"
import { Effect, Schema } from "effect"
import { OpenCodeSavedText, verifyOpenCodeStream } from "../src/harnesses/opencode-stream"

const part = { id: "text-1", sessionID: "session-1", messageID: "message-1", type: "text", text: "", time: { start: 1 } }
const properties = { sessionID: "session-1", time: 1, part }
const ready = { type: "lab.observer.ready" }
const start = { id: "event-1", type: "message.part.updated", properties }
const delta = { id: "event-2", type: "message.part.delta", properties: { sessionID: "session-1", messageID: "message-1", partID: "text-1", field: "text", delta: "HELLO" } }
const end = { id: "event-3", type: "message.part.updated", properties: { ...properties, part: { ...part, text: "HELLO", time: { start: 1, end: 2 } } } }
const json = Schema.encodeSync(Schema.parseJson(Schema.Unknown))
const saved = Schema.decodeUnknownSync(Schema.Array(OpenCodeSavedText))([{ id: "message-1", text: "HELLO" }])
const check = (events: readonly unknown[], text = "HELLO", transcript = saved) => Effect.runPromise(
  verifyOpenCodeStream(events.map(value => json(value)).join("\n") + "\n", "session-1", text, transcript).pipe(Effect.either))

test("native streaming agrees with both the CLI and persisted assistant text", async () => {
  const result = await check([ready, start, delta, end])
  expect(result._tag).toBe("Right")
  if (result._tag === "Right") expect(result.right).toMatchObject({ text: "HELLO", deltas: 1, messageIds: ["message-1"] })
})

test.each([
  ["missing observer", [start, delta, end]],
  ["unfinished stream", [ready, start, delta]],
  ["delta before start", [ready, delta, start, end]],
  ["delta after end", [ready, start, delta, end, { ...delta, id: "event-4" }]],
  ["duplicate event", [ready, start, delta, delta, end]],
  ["restarted part", [ready, start, { ...start, id: "event-4" }, delta, end]],
  ["foreign session", [ready, { ...start, properties: { ...properties, sessionID: "other" } }, delta, end]],
  ["foreign message", [ready, { ...start, properties: { ...properties, part: { ...part, messageID: "other" } } }, delta, end]],
  ["changed final text", [ready, start, delta, { ...end, properties: { ...end.properties, part: { ...end.properties.part, text: "OTHER" } } }]],
  ["completed text without deltas", [ready, start, end]],
  ["overflow", [ready, { type: "lab.observer.overflow" }]],
] as const)("rejects %s", async (_, events) => {
  expect((await check(events))._tag).toBe("Left")
})

test("rejects disagreement with CLI output or persisted text", async () => {
  expect((await check([ready, start, delta, end], "OTHER"))._tag).toBe("Left")
  expect((await check([ready, start, delta, end], "HELLO", [{ ...saved[0]!, text: "OTHER" }]))._tag).toBe("Left")
})

test("rejects a truncated file even when its final JSON object is valid", async () => {
  const result = await Effect.runPromise(verifyOpenCodeStream([ready, start, delta, end].map(value => json(value)).join("\n"), "session-1", "HELLO", saved).pipe(Effect.either))
  expect(result._tag).toBe("Left")
})
