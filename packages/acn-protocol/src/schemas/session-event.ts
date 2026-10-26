import { Schema } from "effect"
import { Millis, optional, tagged, versioned } from "./codec"
import { MessageId } from "./identifier"
import { ServingModelId } from "./serving-usage"

// The observations a session's log is made of (spec section 3.2). These shapes are the stored
// format: a change that old events don't decode under is a new version of that event's type, which
// its `Stored` schema lists.

// Stored events constrain structure only. Limits on what may be requested are command policy and
// live on the command payloads, so changing a limit never invalidates an event already stored.
// JSON cannot hold non-finite numbers.
const finite = Schema.Number.pipe(Schema.finite())
const integer = Schema.Number.pipe(Schema.int())

/** The sampling settings a response was requested with. */
export const ResponseSettings = Schema.Struct({
  systemPrompt: Schema.String,
  temperature: optional(finite),
  maxOutputTokens: optional(integer),
  topP: optional(finite),
  topK: optional(integer),
  minP: optional(finite),
  repetitionPenalty: optional(finite),
  presencePenalty: optional(finite),
  frequencyPenalty: optional(finite),
  seed: optional(integer),
  stop: Schema.Array(Schema.String),
})
export type ResponseSettings = typeof ResponseSettings.Type

/** A reasoning level as requested: the model declares which it offers, so any value is structurally valid. */
export const ResponseReasoningEffort = Schema.String
export type ResponseReasoningEffort = typeof ResponseReasoningEffort.Type

export const ResponseOutputPart = Schema.Literal("text", "reasoning")
export type ResponseOutputPart = typeof ResponseOutputPart.Type

/** Why a response failed: each way a response fails, with what it tells. */
export const ResponseFailure = Schema.Union(
  /** The inference service's error, as it reported it. */
  tagged("inference", { code: Schema.String, message: Schema.String, param: optional(Schema.String) }),
  /** The inference service could not be reached, or its stream broke or was malformed. */
  tagged("unreachable", { message: Schema.String }),
  /** The request was refused as invalid before it was sent. */
  tagged("rejected", { message: Schema.String }),
  /** The response's output could not be recorded. */
  tagged("storage", { message: Schema.String }),
  /** The stream finished without its timings report. */
  tagged("protocol", { message: Schema.String })
)
export type ResponseFailure = typeof ResponseFailure.Type

export const ResponseOutcome = Schema.Union(
  /** `finishReason` is the engine's `finish_reason`, verbatim. */
  tagged("finished", { finishReason: Schema.String }),
  tagged("failed", { error: ResponseFailure }),
  tagged("stopped", { by: Schema.Literal("user", "shutdown") })
)
export type ResponseOutcome = typeof ResponseOutcome.Type

/**
 * What the engine counted of a response, from its timings report: the prompt's tokens, cached ones
 * included, those found in the cache, those generated, and the speculative drafts when it drafted.
 */
export const ResponseUsage = Schema.Struct({
  promptTokens: Schema.NonNegativeInt,
  cachedTokens: Schema.NonNegativeInt,
  completionTokens: Schema.NonNegativeInt,
  draftTokens: optional(Schema.NonNegativeInt),
  draftAcceptedTokens: optional(Schema.NonNegativeInt),
})
export type ResponseUsage = typeof ResponseUsage.Type

/** How long the engine spent on a response, from the same report: durations only. */
export const ResponseTimings = Schema.Struct({
  promptMs: finite,
  predictedMs: finite,
  samplerMs: finite,
  parserMs: finite,
  timeToFirstTokenMs: finite,
})
export type ResponseTimings = typeof ResponseTimings.Type

// Every event is observed at one time. Its session is the log that holds it.
const envelope = { time: Millis }

export const MessageSent = tagged("message_sent", {
  ...envelope,
  messageId: MessageId,
  text: Schema.String,
})
export type MessageSent = typeof MessageSent.Type

export const ResponseStarted = tagged("response_started", {
  ...envelope,
  responseId: MessageId,
  /** The response this one replaces (retry). */
  replaces: optional(MessageId),
  /** The servable model that was requested. */
  model: ServingModelId,
  reasoningEffort: optional(ResponseReasoningEffort),
  settings: ResponseSettings,
})
export type ResponseStarted = typeof ResponseStarted.Type

/** The deltas of one part, each exactly as received, with its arrival in ms after `time`. */
export const ResponseOutput = tagged("response_output", {
  ...envelope,
  responseId: MessageId,
  part: ResponseOutputPart,
  deltas: Schema.NonEmptyArray(Schema.String),
  at: Schema.NonEmptyArray(Schema.NonNegativeInt),
}).pipe(
  Schema.filter((event) =>
    event.at.length !== event.deltas.length
      ? "Every delta has exactly one arrival time"
      : event.at[0] === 0 || "The first delta arrives at the event's time"
  )
)
export type ResponseOutput = typeof ResponseOutput.Type

export const ResponseEnded = tagged("response_ended", {
  ...envelope,
  responseId: MessageId,
  outcome: ResponseOutcome,
  /**
   * The engine's last timings report, both or neither: present once the response reached the model
   * and the engine reported, so always for a finished one, and absent for one ended while queued,
   * refused before generating, or recovered after a crash.
   */
  usage: optional(ResponseUsage),
  timings: optional(ResponseTimings),
})
export type ResponseEnded = typeof ResponseEnded.Type

export const SessionEvent = Schema.Union(MessageSent, ResponseStarted, ResponseOutput, ResponseEnded)
export type SessionEvent = typeof SessionEvent.Type

// Each type's stored versions. A breaking change to a type adds its new version here and lists the
// previous one, with a schema that decodes it to the current shape (spec section 8).
export const StoredMessageSent = versioned({ version: 1, schema: MessageSent })
export const StoredResponseStarted = versioned({ version: 1, schema: ResponseStarted })
export const StoredResponseOutput = versioned({ version: 1, schema: ResponseOutput })
export const StoredResponseEnded = versioned({ version: 1, schema: ResponseEnded })
