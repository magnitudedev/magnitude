import { Array as Arr, DateTime, HashMap, Option, Schema } from "effect"
import { Projection } from "@magnitudedev/event-engine"
import {
  appendOutput,
  type ChatAssistantMessage,
  ChatMessage,
  latestResponse,
  LiveResponse,
  MessageId,
  type ResponseEnded,
} from "@magnitudedev/acn-protocol"
import { ChatEngine, ResponseActivity } from "./engine"

// A session's folds (spec section 4): pure and synchronous, each over its own state. None is
// stored: each is what the session's log replays to.

/** What an ID the session used names: a user message, or a response to one. */
export const SeenId = Schema.Union(
  Schema.TaggedStruct("message", {}),
  Schema.TaggedStruct("response", { message: MessageId })
)
export type SeenId = typeof SeenId.Type

export const SeenState = Schema.Struct({
  ids: Schema.HashMap({ key: MessageId, value: SeenId }),
  /** The latest user message, which a new response answers. */
  latest: Schema.Option(MessageId),
})
export type SeenState = typeof SeenState.Type

/**
 * Every message and response ID the session has used, with what it names, so a replayed command
 * finds its own effect. A response answers the latest message: a retry replaces the latest
 * response, which answers it too.
 */
export const Seen = ChatEngine.Projection.singleton({
  name: "Seen",
  state: SeenState,
  initial: { ids: HashMap.empty(), latest: Option.none() } as SeenState,
  handlers: (on) => [
    on.event("message_sent", ({ event, state }) => Projection.set<SeenState>({
      ids: HashMap.set(state.ids, event.messageId, { _tag: "message" }),
      latest: Option.some(event.messageId),
    })),
    // A response always follows the message it answers.
    on.event("response_started", ({ event, state }) =>
      Option.match(state.latest, {
        onNone: () => Projection.keep,
        onSome: (message) => Projection.set<SeenState>({ ...state, ids: HashMap.set(state.ids, event.responseId, { _tag: "response", message }) }),
      })),
  ],
})

/** Whether a message is this response. */
export const isResponse = (responseId: MessageId) => (message: ChatMessage): message is ChatAssistantMessage =>
  message._tag === "assistant" && message.id === responseId

/** A response as its end leaves it. */
export const withEnd = (message: ChatAssistantMessage, event: ResponseEnded): ChatAssistantMessage => ({
  ...message,
  end: Option.some({
    time: event.time,
    outcome: event.outcome,
    usage: event.usage,
    timings: event.timings,
  }),
})

/** Updates the transcript's message for this response; every other message keeps its identity. */
const updateResponse = (
  transcript: ReadonlyArray<ChatMessage>,
  responseId: MessageId,
  update: (message: ChatAssistantMessage) => ChatAssistantMessage
) => {
  const index = transcript.findLastIndex(isResponse(responseId))
  const message = transcript[index]
  return message?._tag === "assistant"
    ? Projection.set(Arr.replace(transcript, index, update(message)))
    : Projection.keep
}

/**
 * Every user message and the latest attempt of each response, in order. Its latest response is the
 * one a retry may replace, and is running until it has an end.
 */
export const Transcript = ChatEngine.Projection.singleton({
  name: "Transcript",
  state: Schema.Array(ChatMessage),
  initial: [] as ReadonlyArray<ChatMessage>,
  handlers: (on) => [
    on.event("message_sent", ({ event, state }) =>
      Projection.set<ReadonlyArray<ChatMessage>>(Arr.append(state, { _tag: "user", id: event.messageId, text: event.text, time: event.time }))),
    on.event("response_started", ({ event, state }) => {
      const message: ChatMessage = {
        _tag: "assistant",
        id: event.responseId,
        model: event.model,
        reasoningEffort: event.reasoningEffort,
        settings: event.settings,
        started: event.time,
        content: [],
        end: Option.none(),
      }
      // A retry replaces the latest response, which is the last message.
      return Projection.set<ReadonlyArray<ChatMessage>>(
        Arr.append(Option.isSome(event.replaces) ? state.slice(0, -1) : state, message)
      )
    }),
    on.event("response_output", ({ event, state }) =>
      updateResponse(state, event.responseId, (message) => ({
        ...message,
        content: appendOutput(message.content, {
          _tag: event.part,
          text: event.deltas.join(""),
          time: { started: event.time, ended: DateTime.add(event.time, { millis: Arr.lastNonEmpty(event.at) }) },
        }),
      }))),
    on.event("response_ended", ({ event, state }) =>
      updateResponse(state, event.responseId, (message) => withEnd(message, event))),
  ],
})

/** The chat's running response, if one is running. */
export const runningResponse = (transcript: ReadonlyArray<ChatMessage>): Option.Option<MessageId> =>
  Option.flatMap(latestResponse(transcript), (response) => Option.isNone(response.end) ? Option.some(response.id) : Option.none())

const updateLive = (
  state: Option.Option<LiveResponse>,
  responseId: MessageId,
  update: (live: LiveResponse) => LiveResponse
) =>
  Option.match(Option.filter(state, (live) => live.id === responseId), {
    onNone: () => Projection.keep,
    onSome: (live) => Projection.set(Option.some(update(live))),
  })

/**
 * The running response's live state: its phase and latest timings, and the output streamed since
 * its last recorded group. Recording a group moves its output from here to the transcript in the
 * same settlement, and ending the response clears this in the settlement that records the end. A
 * watcher reads both in one step and receives each settlement as one update, which a client applies
 * in one step, so a client never shows output twice or misses any.
 */
export const Live = ChatEngine.Projection.singleton({
  name: "Live",
  state: Schema.Option(LiveResponse),
  initial: Option.none<LiveResponse>(),
  handlers: (on) => [
    on.event("response_started", ({ event }) =>
      Projection.set(Option.some<LiveResponse>({
        id: event.responseId,
        phase: Option.none(),
        usage: Option.none(),
        timings: Option.none(),
        pending: Option.none(),
      }))),
    on.ambient(ResponseActivity, ({ value: latest, state }) =>
      Option.match(latest, {
        onNone: () => Projection.keep,
        onSome: (value) => updateLive(state, value.responseId, (live) => {
        switch (value._tag) {
          case "phase":
            return { ...live, phase: Option.some(value.phase) }
          case "report":
            return { ...live, usage: Option.some(value.usage), timings: Option.some(value.timings) }
          case "output":
            return {
              ...live,
              pending: Option.some(Option.match(live.pending, {
                onNone: () => value.part,
                onSome: (open) => Arr.lastNonEmpty(appendOutput([open], value.part)),
              })),
            }
        }
      }),
      })),
    on.event("response_output", ({ event, state }) =>
      updateLive(state, event.responseId, (live) => ({ ...live, pending: Option.none() }))),
    on.event("response_ended", ({ event, state }) =>
      Option.exists(state, (live) => live.id === event.responseId) ? Projection.set(Option.none()) : Projection.keep),
  ],
})

/**
 * The last moment a running response was known to be alive (DESIGN A2): the later of its start and
 * the end of its last recorded output. Nothing but its output joins the log while it runs.
 */
export const lastObserved = (response: ChatAssistantMessage): DateTime.Utc =>
  Option.match(Arr.last(response.content), {
    onNone: () => response.started,
    onSome: (part) => DateTime.max(response.started, part.time.ended),
  })
