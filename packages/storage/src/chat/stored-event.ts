import { ParseResult, Schema } from "effect"
import {
  SessionEvent,
  StoredMessageSent,
  StoredResponseEnded,
  StoredResponseOutput,
  StoredResponseStarted,
} from "@magnitudedev/acn-protocol"
import { Fields, FieldsWithout } from "./stored-fields"

// The only mapping between an `events` row and a session event. The row's `type` and `version`
// columns own the event's type and its schema version, and `created_at` owns the event's `time`;
// `data` holds the rest of the encoded event and never repeats them. The session is the log the
// row belongs to, so no event names it.

type Versioned = { readonly version: number; readonly value: unknown }

// Each type's versioned schema: it encodes the current version and decodes every version it lists.
const stored: { readonly [Type in SessionEvent["_tag"]]: Schema.Schema<Extract<SessionEvent, { _tag: Type }>, Versioned> } = {
  message_sent: StoredMessageSent,
  response_started: StoredResponseStarted,
  response_output: StoredResponseOutput,
  response_ended: StoredResponseEnded,
}
const isType = (type: string): type is SessionEvent["_tag"] => Object.hasOwn(stored, type)
const schemaOf = (type: SessionEvent["_tag"]) => stored[type] as Schema.Schema<SessionEvent, Versioned>

const EventRow = Schema.Struct({
  type: Schema.String,
  version: Schema.Number,
  created_at: Schema.Number,
  data: Schema.parseJson(FieldsWithout("type", "version", "time")),
})

const Encoded = Schema.Struct({ type: Schema.String, time: Schema.Number }, Fields)

export const StoredEvent = Schema.transformOrFail(EventRow, Schema.typeSchema(SessionEvent), {
  strict: true,
  decode: ({ type, version, created_at, data }, _, ast) =>
    isType(type)
      ? ParseResult.decodeUnknown(schemaOf(type), { onExcessProperty: "error" })({ version, value: { ...data, type, time: created_at } })
      : ParseResult.fail(new ParseResult.Type(ast, type, `Event type ${type} is not supported`)),
  encode: (event) => ParseResult.flatMap(ParseResult.encode(schemaOf(event._tag))(event), ({ version, value }) =>
    ParseResult.map(ParseResult.decodeUnknown(Encoded)(value), ({ type, time, ...data }) => ({ type, version, created_at: time, data }))),
})
