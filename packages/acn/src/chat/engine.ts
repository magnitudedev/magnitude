import { Effect, Option } from "effect"
import { Ambient, EventEngine, type EventSettlementError, isEventSettlementError } from "@magnitudedev/event-engine"
import {
  type ChatContentPart,
  ChatStorageFailed,
  type LivePhase,
  type MessageId,
  type ResponseTimings,
  type ResponseUsage,
  type SessionEvent,
} from "@magnitudedev/acn-protocol"
import type { ChatStoreError } from "@magnitudedev/storage"

/** The chat engine family: one session's events, stored in the chat store. */
export const ChatEngine = EventEngine.define<SessionEvent, "_tag", ChatStoreError>({ discriminator: "_tag" })

/** What a running response's stream shows that is live state only: never stored or replayed. */
export type ResponseActivity =
  | { readonly _tag: "phase"; readonly responseId: MessageId; readonly phase: LivePhase }
  | { readonly _tag: "output"; readonly responseId: MessageId; readonly part: ChatContentPart }
  | { readonly _tag: "report"; readonly responseId: MessageId; readonly usage: ResponseUsage; readonly timings: ResponseTimings }

/** The latest activity of the running response; none until its stream shows any. */
export const ResponseActivity = Ambient.define<Option.Option<ResponseActivity>>({
  name: "ResponseActivity",
  initial: Option.none(),
})

/**
 * Settles an effect over the chat engine's state. Every chat projection read names a projection of
 * the chat engine and no chat projection emits signals, so a settlement error is a defect.
 */
export const settled = <A, E, R>(effect: Effect.Effect<A, E, R>) =>
  Effect.catchIf(effect, (error): error is Extract<E, EventSettlementError> => isEventSettlementError(error), Effect.die)

/** What a client is told when the chat store fails. */
export const storageFailed = (error: ChatStoreError): ChatStorageFailed => {
  switch (error._tag) {
    case "ChatStoreUnavailable":
      return new ChatStorageFailed({ reason: "unavailable" })
    case "ChatStoreCorrupt":
      return new ChatStorageFailed({ reason: "unreadable" })
    case "ChatStoreIncompatible":
      return new ChatStorageFailed({ reason: error.newer ? "newer_version" : "unrecognized_format" })
    case "ChatStoreRejected":
      return new ChatStorageFailed({ reason: "unwritable" })
  }
}
