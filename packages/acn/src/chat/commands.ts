import { DateTime, Effect, Either, HashMap, Option } from "effect"
import type { ProjectionReader, ProjectionReadError, Transaction } from "@magnitudedev/event-engine"
import {
  ChatBusy,
  type ChatMessagePayload,
  ChatNotFound,
  type ChatRetryPayload,
  type ChatStopPayload,
  type ChatMessage,
  InvalidChatRequest,
  latestResponse,
  type MessageId,
  type MessageSent,
  type ResponseStarted,
  type SessionEvent,
} from "@magnitudedev/acn-protocol"
import { runningResponse, Seen, type SeenId, type SeenState, Transcript } from "./folds"
import { Generation } from "./generation"

// The chat commands (spec section 5), as transaction bodies: each reads settled state, decides, and
// emits in one step under the engine's acceptance lock. A rejection commits nothing, and a command
// that already took effect, replayed, does nothing.

type Decision<Rejection> = Transaction<SessionEvent, Either.Either<void, Rejection>>

const accept = (events: ReadonlyArray<SessionEvent>): Decision<never> => ({ events, result: Either.void })
const reject = <Rejection>(rejection: Rejection): Decision<Rejection> => ({ events: [], result: Either.left(rejection) })

/**
 * The state a command decides on: whether its session exists, and the session's settled folds. A
 * session exists once its log holds anything; a log begins with its first message, so it does
 * exactly when its transcript holds a message.
 */
export interface States {
  readonly exists: boolean
  readonly seen: SeenState
  readonly transcript: ReadonlyArray<ChatMessage>
}

/** A command's body, at the time it is decided. */
export type Command<Rejection> = (read: ProjectionReader) => Effect.Effect<Decision<Rejection>, ProjectionReadError>

/** Reads what commands decide on. */
export const states = (read: ProjectionReader): Effect.Effect<States, ProjectionReadError> =>
  Effect.map(Effect.all({ seen: read.get(Seen), transcript: read.get(Transcript) }), ({ seen, transcript }) => ({
    exists: transcript.length > 0,
    seen,
    transcript,
  }))

const decided = <Rejection>(decide: (states: States, time: DateTime.Utc) => Decision<Rejection>): Command<Rejection> =>
  (read) => Effect.map(Effect.all([states(read), DateTime.now]), ([current, time]) => decide(current, time))

const idsInUse = new InvalidChatRequest({ reason: "ids_in_use" })

/** What the session used this ID for, if it used it. */
const named = (states: States, id: MessageId): Option.Option<SeenId> => HashMap.get(states.seen.ids, id)

/** The message this ID's response answers, if it names a response. */
const answered = (states: States, id: MessageId): Option.Option<MessageId> =>
  Option.flatMap(named(states, id), (used) => used._tag === "response" ? Option.some(used.message) : Option.none())

/** A command's IDs: all unused, used by this very command, which is replayed, or used otherwise. */
type Use = "unused" | "replayed" | "in_use"

/** A send is replayed only when its response answers its message. */
const sendUse = (states: States, input: ChatMessagePayload): Use =>
  Option.contains(answered(states, input.responseId), input.messageId)
    ? "replayed"
    : Option.isNone(named(states, input.messageId)) && Option.isNone(named(states, input.responseId))
    ? "unused"
    : "in_use"

/** A retry is replayed only when its response answers the message of the response it replaces. */
const retryUse = (states: States, input: ChatRetryPayload): Use =>
  Option.isNone(named(states, input.responseId))
    ? "unused"
    : Option.exists(Option.all([answered(states, input.responseId), answered(states, input.replaces)]), ([own, replaced]) => own === replaced)
    ? "replayed"
    : "in_use"

/** A start is replayed only when its message is the session's first and its response answers it. */
const startUse = (states: States, input: ChatMessagePayload): Use =>
  !states.exists
    ? "unused"
    : states.transcript[0]?.id === input.messageId && sendUse(states, input) === "replayed"
    ? "replayed"
    : "in_use"

const messageSent = (input: ChatMessagePayload, time: DateTime.Utc): MessageSent => ({
  _tag: "message_sent",
  time,
  messageId: input.messageId,
  text: input.text,
})

const responseStarted = (
  input: ChatMessagePayload | ChatRetryPayload,
  replaces: Option.Option<MessageId>,
  time: DateTime.Utc
): ResponseStarted => ({
  _tag: "response_started",
  time,
  responseId: input.responseId,
  replaces,
  model: input.model,
  reasoningEffort: input.reasoningEffort,
  settings: input.settings,
})

/** Whether a start already took effect, which makes model validation unnecessary. */
export const startReplayed = (input: ChatMessagePayload) => (states: States) => startUse(states, input) === "replayed"

/** Whether a send already took effect, which makes model validation unnecessary. */
export const sendReplayed = (input: ChatMessagePayload) => (states: States) => sendUse(states, input) === "replayed"

/** Whether a retry already took effect. */
export const retryReplayed = (input: ChatRetryPayload) => (states: States) => retryUse(states, input) === "replayed"

/**
 * Starts a chat with its first message: its first events, whose commit creates the session's record.
 * Only a session that does not exist starts; one this start already started is left as it is.
 */
export const start = (input: ChatMessagePayload): Command<InvalidChatRequest> =>
  decided<InvalidChatRequest>((states, time) => {
    switch (startUse(states, input)) {
      case "replayed":
        return accept([])
      case "in_use":
        return reject(idsInUse)
      case "unused":
        return accept([messageSent(input, time), responseStarted(input, Option.none(), time)])
    }
  })

export const send = (input: ChatMessagePayload): Command<ChatNotFound | ChatBusy | InvalidChatRequest> =>
  decided<ChatNotFound | ChatBusy | InvalidChatRequest>((states, time) => {
    if (!states.exists) return reject(new ChatNotFound({ chatId: input.chatId }))
    switch (sendUse(states, input)) {
      case "replayed":
        return accept([])
      case "in_use":
        return reject(idsInUse)
      case "unused":
        return Option.isSome(runningResponse(states.transcript))
          ? reject(new ChatBusy({ chatId: input.chatId }))
          : accept([messageSent(input, time), responseStarted(input, Option.none(), time)])
    }
  })

/** Replaces the latest response, once it has ended, with a new one. */
export const retry = (input: ChatRetryPayload): Command<ChatNotFound | ChatBusy | InvalidChatRequest> =>
  decided<ChatNotFound | ChatBusy | InvalidChatRequest>((states, time) => {
    if (!states.exists) return reject(new ChatNotFound({ chatId: input.chatId }))
    switch (retryUse(states, input)) {
      case "replayed":
        return accept([])
      case "in_use":
        return reject(idsInUse)
      case "unused":
        return Option.isSome(runningResponse(states.transcript))
          ? reject(new ChatBusy({ chatId: input.chatId }))
          : Option.exists(latestResponse(states.transcript), (latest) => latest.id === input.replaces)
          ? accept([responseStarted(input, Option.some(input.replaces), time)])
          : reject(new InvalidChatRequest({ reason: "not_latest_response" }))
    }
  })

/**
 * Stops the named response only if it is the one running, in the same step that checks it, so a
 * stop can never reach a later response. Its generation records the end.
 */
export const stop = ({ chatId, responseId }: ChatStopPayload): Command<ChatNotFound | InvalidChatRequest> =>
  decided<ChatNotFound | InvalidChatRequest>((states) => {
    if (!states.exists) return reject(new ChatNotFound({ chatId }))
    if (Option.isNone(answered(states, responseId))) return reject(new InvalidChatRequest({ reason: "no_such_response" }))
    return Option.contains(runningResponse(states.transcript), responseId)
      ? { events: [], interrupts: [Generation], result: Either.void }
      : accept([])
  })
