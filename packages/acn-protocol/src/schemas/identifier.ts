import { Effect, Schema } from "effect"
import { createId } from "@magnitudedev/generate-id"

// IDs come from the repo's generator (cuid2, 12 characters) and carry no order: transcripts order
// by log position and sessions by activity. Stored IDs are checked for structure only, so the
// generator can change without invalidating stored events.

export const SessionId = Schema.NonEmptyString.pipe(Schema.brand("SessionId"))
export type SessionId = typeof SessionId.Type

/** A user message or a response, since a response is an assistant message. */
export const MessageId = Schema.NonEmptyString.pipe(Schema.brand("MessageId"))
export type MessageId = typeof MessageId.Type

export const generateSessionId: Effect.Effect<SessionId> = Effect.sync(() => SessionId.make(createId()))

export const generateMessageId: Effect.Effect<MessageId> = Effect.sync(() => MessageId.make(createId()))
