import { Array as Arr, Option, Schema } from "effect"
import { Millis, optional, tagged } from "./codec"
import { SessionId } from "./identifier"
import { appendOutput, type ChatAssistantMessage, ChatMessage, LiveResponse } from "./chat-message"

// A chat as the chat list and its watchers see it (spec section 4): its session's record, and
// whether a response is running.

export const ChatSummary = Schema.Struct({
  /** A chat is a session: its identity is the session's. */
  id: SessionId,
  /** The title chosen by rename, else the first message's text. */
  title: Schema.String,
  /** The session's creation, and the later of its latest event other than a response's output and its latest rename. */
  time: Schema.Struct({ created: Millis, updated: Millis }),
  /** The messages its transcript shows: every user message and the latest attempt of each response. */
  messageCount: Schema.NonNegativeInt,
  /** Whether a response is running: one runs only while its chat is open in the service. */
  running: Schema.Boolean,
})
export type ChatSummary = typeof ChatSummary.Type

/** Chats by latest activity, most recent first. */
export const ChatPage = Schema.Struct({
  chats: Schema.Array(ChatSummary),
  hasMore: Schema.Boolean,
})
export type ChatPage = typeof ChatPage.Type

/**
 * What a chat's watcher receives: the chat as it is, then one `changed` per settlement it observes,
 * carrying every part that settlement changed, until it is removed. A `changed` holds the summary if
 * it changed, each transcript message that changed (set at its index, or appended at the end), and
 * the running response's live state if it changed, so a client applies a settlement in one step.
 */
export const ChatUpdate = Schema.Union(
  tagged("snapshot", { chat: ChatSummary, messages: Schema.Array(ChatMessage), live: optional(LiveResponse) }),
  tagged("changed", {
    chat: optional(ChatSummary),
    messages: Schema.Array(Schema.Struct({ index: Schema.NonNegativeInt, message: ChatMessage })),
    /** Present when the live state changed: `null` when no response is running. */
    live: optional(Schema.OptionFromNullOr(LiveResponse)),
  }),
  tagged("removed", {})
)
export type ChatUpdate = typeof ChatUpdate.Type

/** A watched chat as its watch last described it: its folded state and running response, or its removal. */
export type WatchedChat =
  | {
      readonly _tag: "open"
      readonly chat: ChatSummary
      readonly messages: ReadonlyArray<ChatMessage>
      readonly live: Option.Option<LiveResponse>
    }
  | { readonly _tag: "removed" }

type OpenChat = Extract<WatchedChat, { _tag: "open" }>

/** A watch that breaks its own order: a defect of the server or the transport, never a state of the chat. */
const outOfOrder = (what: string): never => {
  throw new Error(`A chat watch ${what}.`)
}

const opened = (previous: Option.Option<WatchedChat>): OpenChat =>
  Option.match(previous, {
    onNone: () => outOfOrder("began without its snapshot"),
    onSome: (watched) => watched._tag === "open" ? watched : outOfOrder("continued after the chat's removal"),
  })

/** Sets a message at its index, or appends it at the end. */
const setMessage = (messages: ReadonlyArray<ChatMessage>, { index, message }: { readonly index: number; readonly message: ChatMessage }) =>
  index < messages.length
    ? Arr.replace(messages, index, message)
    : index === messages.length
    ? Arr.append(messages, message)
    : outOfOrder(`set message ${index} of ${messages.length}`)

/** Applies one update: a snapshot replaces the chat, and a change applies a whole settlement in one step. */
export const applyChatUpdate = (previous: Option.Option<WatchedChat>, update: ChatUpdate): WatchedChat => {
  switch (update._tag) {
    case "snapshot":
      // A reconnected watch begins again with a snapshot.
      if (Option.exists(previous, (watched) => watched._tag === "removed")) return outOfOrder("continued after the chat's removal")
      return { _tag: "open", chat: update.chat, messages: update.messages, live: update.live }
    case "changed": {
      const current = opened(previous)
      return {
        _tag: "open",
        chat: Option.getOrElse(update.chat, () => current.chat),
        messages: update.messages.reduce(setMessage, current.messages),
        live: Option.getOrElse(update.live, () => current.live),
      }
    }
    case "removed":
      opened(previous)
      return { _tag: "removed" }
  }
}

/** The message's response as it runs: its phase and the engine's latest measurements, once the engine has said. */
export const liveResponse = (watched: OpenChat, message: ChatAssistantMessage): Option.Option<LiveResponse> =>
  Option.filter(watched.live, (response) => response.id === message.id)

/** The transcript, its running response showing the output streamed since its last recorded group. */
export const chatMessages = (watched: OpenChat): ReadonlyArray<ChatMessage> =>
  watched.messages.map((message) =>
    message._tag === "user"
      ? message
      : Option.match(Option.flatMap(liveResponse(watched, message), (response) => response.pending), {
          onNone: () => message,
          onSome: (pending) => ({ ...message, content: appendOutput(message.content, pending) }),
        })
  )

// Each error owns its text, so the server and every client say the same thing.

export class ChatNotFound extends Schema.TaggedError<ChatNotFound>()("ChatNotFound", {
  chatId: SessionId,
}) {
  get message() {
    return "This conversation no longer exists."
  }
}

/** A response is running in the chat. */
export class ChatBusy extends Schema.TaggedError<ChatBusy>()("ChatBusy", {
  chatId: SessionId,
}) {
  get message() {
    return "A response is already running in this conversation."
  }
}

export const InvalidChatReason = Schema.Literal(
  "model_unavailable",
  "reasoning_unsupported",
  "ids_in_use",
  "not_latest_response",
  "no_such_response",
  "title_changed"
)
export type InvalidChatReason = typeof InvalidChatReason.Type

const invalidRequests: { readonly [Reason in InvalidChatReason]: string } = {
  model_unavailable: "This model is unavailable for text chat. Choose an installed text model.",
  reasoning_unsupported: "This model does not support that reasoning level.",
  ids_in_use: "These IDs were already used for other messages.",
  not_latest_response: "Only the last response can be retried.",
  no_such_response: "This chat has no such response.",
  title_changed: "This conversation was renamed meanwhile. Rename it again from its current title.",
}

export class InvalidChatRequest extends Schema.TaggedError<InvalidChatRequest>()("InvalidChatRequest", {
  reason: InvalidChatReason,
}) {
  get message() {
    return invalidRequests[this.reason]
  }
}

export const ChatStorageFailure = Schema.Literal("unavailable", "unreadable", "unwritable", "newer_version", "unrecognized_format")
export type ChatStorageFailure = typeof ChatStorageFailure.Type

const storageFailures: { readonly [Failure in ChatStorageFailure]: string } = {
  unavailable: "Chat history could not be read or saved. Check access to the Magnitude data folder.",
  unreadable: "Chat history could not be read.",
  unwritable: "This change could not be saved to chat history.",
  newer_version: "Chat history was saved by a newer Magnitude version. Update Magnitude to open it.",
  unrecognized_format: "The chat history file is not in a format Magnitude recognizes.",
}

/** Chats could not be read or written. */
export class ChatStorageFailed extends Schema.TaggedError<ChatStorageFailed>()("ChatStorageFailed", {
  reason: ChatStorageFailure,
}) {
  get message() {
    return storageFailures[this.reason]
  }
}
