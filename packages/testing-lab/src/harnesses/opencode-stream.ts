import { defineFSM } from "@magnitudedev/utils/fsm"
import { Effect, Option, Schema } from "effect"
import { AssertionFailure } from "../domain"

const SessionId = Schema.NonEmptyString.pipe(Schema.brand("OpenCodeSessionId"))
const MessageId = Schema.NonEmptyString.pipe(Schema.brand("OpenCodeMessageId"))
const TextId = Schema.NonEmptyString.pipe(Schema.brand("OpenCodeTextId"))
const EventId = Schema.NonEmptyString.pipe(Schema.brand("OpenCodeEventId"))
const Part = Schema.Struct({ id: TextId, messageID: MessageId, sessionID: SessionId, type: Schema.Literal("text"), text: Schema.String,
  time: Schema.Struct({ start: Schema.Number, end: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }) }) })
export const OpenCodeTextEvent = Schema.Union(
  Schema.Struct({ id: EventId, type: Schema.Literal("message.part.updated"), properties: Schema.Struct({ sessionID: SessionId, part: Part, time: Schema.Number }) }),
  Schema.Struct({ id: EventId, type: Schema.Literal("message.part.delta"), properties: Schema.Struct({ sessionID: SessionId, messageID: MessageId, partID: TextId, field: Schema.Literal("text"), delta: Schema.String }) }),
)
export const OpenCodeStreamReceipt = Schema.Struct({ sessionId: SessionId, messageIds: Schema.NonEmptyArray(MessageId),
  text: Schema.NonEmptyString, deltas: Schema.Int.pipe(Schema.positive()) })
export const OpenCodeSavedText = Schema.Struct({ id: MessageId, text: Schema.String })
const textFields = { text: Schema.String, deltas: Schema.Int.pipe(Schema.nonNegative()) }
class Streaming extends Schema.TaggedClass<Streaming>()("Streaming", textFields) {}
class Completed extends Schema.TaggedClass<Completed>()("Completed", textFields) {}
const textFSM = defineFSM({ Streaming, Completed }, { Streaming: ["Streaming", "Completed"], Completed: [] })
const fail = (message: string) => new AssertionFailure({ message })

/** The pinned client's own lifecycle events must agree with its CLI output and persisted messages. */
export const verifyOpenCodeStream = (jsonl: string, session: string, text: string, saved: readonly typeof OpenCodeSavedText.Type[]) => Effect.gen(function* () {
  if (Buffer.byteLength(jsonl) > 4 * 1024 * 1024 || !jsonl.endsWith("\n")) return yield* fail("OpenCode streaming evidence is oversized or truncated")
  const lines = jsonl.split("\n").filter(Boolean)
  yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ type: Schema.Literal("lab.observer.ready") })))(lines.shift()).pipe(
    Effect.mapError(() => fail("OpenCode did not initialize its native event observer")))
  const sessionId = yield* Schema.decodeUnknown(SessionId)(session)
  const messages = new Map<typeof MessageId.Type, Map<typeof TextId.Type, Streaming | Completed>>()
  const seen = new Set<typeof EventId.Type>()
  for (const line of lines) {
    const event = yield* Schema.decodeUnknown(Schema.parseJson(OpenCodeTextEvent))(line).pipe(
      Effect.mapError(() => fail("OpenCode native text event is malformed or the observer overflowed")))
    if (seen.has(event.id)) return yield* fail("OpenCode repeated a native event identity")
    seen.add(event.id)
    const identity = event.type === "message.part.updated" ? event.properties.part : { ...event.properties, id: event.properties.partID }
    if (identity.sessionID !== sessionId || event.properties.sessionID !== sessionId || !saved.some(message => message.id === identity.messageID)) return yield* fail("OpenCode streaming evidence belongs to another session or assistant message")
    let parts = messages.get(identity.messageID)
    if (!parts) { parts = new Map(); messages.set(identity.messageID, parts) }
    const current = parts.get(identity.id)
    if (event.type === "message.part.updated" && Option.isNone(event.properties.part.time.end)) {
      if (current || event.properties.part.text !== "") return yield* fail("OpenCode restarted an existing native text part or omitted its beginning")
      parts.set(identity.id, new Streaming({ text: "", deltas: 0 }))
    } else {
      if (!current || current._tag !== "Streaming") return yield* fail("OpenCode emitted text outside its streaming lifecycle")
      if (event.type === "message.part.delta") {
        parts.set(identity.id, textFSM.transition(current, "Streaming", { text: current.text + event.properties.delta, deltas: current.deltas + (event.properties.delta ? 1 : 0) }))
      } else {
        if (current.text !== event.properties.part.text) return yield* fail("OpenCode final text differs from its observed deltas")
        parts.set(identity.id, textFSM.transition(current, "Completed", { text: current.text, deltas: current.deltas }))
      }
    }
  }
  let observed = "", deltas = 0
  for (const [id, parts] of messages) {
    if ([...parts.values()].some(part => part._tag !== "Completed")) return yield* fail("OpenCode left a native text stream unfinished")
    const value = [...parts.values()].map(part => part.text).join("")
    if (saved.find(message => message.id === id)?.text !== value) return yield* fail("OpenCode persisted text differs from its observed stream")
    observed += value
    deltas += [...parts.values()].reduce((sum, part) => sum + part.deltas, 0)
  }
  if (!observed.trim() || observed !== text || deltas === 0) return yield* fail("OpenCode produced no matching native text stream")
  return yield* Schema.decodeUnknown(OpenCodeStreamReceipt)({ sessionId, messageIds: [...messages.keys()], text: observed, deltas })
})

/** Observation-only OpenCode plugin: synchronous writes preserve native callback order, without altering output. */
export const openCodeStreamObserver = `import { openSync, writeSync, closeSync } from "node:fs";
export default async () => {
  const file = process.env.LAB_OPENCODE_STREAM_LOG;
  if (!file) return {};
  const fd = openSync(file, "wx", 0o600);
  writeSync(fd, JSON.stringify({type:"lab.observer.ready"})+"\\n");
  let bytes = 0, overflowed = false;
  const textParts = new Set();
  return {
    event: ({event}) => {
      if (overflowed) return;
      if (event.type === "message.part.updated") {
        const part = event.properties.part;
        if (part.type !== "text" || typeof part.time?.start !== "number") return;
        textParts.add(part.id);
      } else if (event.type !== "message.part.delta" || event.properties.field !== "text" || !textParts.has(event.properties.partID)) return;
      const line = JSON.stringify(event)+"\\n";
      bytes += Buffer.byteLength(line);
      if (bytes > 4*1024*1024-1024) { overflowed = true; writeSync(fd, JSON.stringify({type:"lab.observer.overflow"})+"\\n"); return; }
      writeSync(fd, line);
    },
    dispose: () => closeSync(fd)
  };
};
`
