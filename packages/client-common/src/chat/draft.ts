import { Atom } from "@effect-atom/atom-react"
import { Effect, Option, Schema } from "effect"
import {
  type ChatAssistantMessage,
  type ChatMessagePayload,
  type ChatMessage,
  type ChatRetryPayload,
  generateMessageId,
  latestResponse,
  type MessageId,
  RequestedResponseSettings,
  type ResponseReasoningEffort,
  responseStatus,
  type ServingModelId,
  type SessionId,
} from "@magnitudedev/sdk"

/** What the next request asks for: a model, its reasoning level and the sampling settings. */
export type ChatRequestSettings = RequestedResponseSettings & {
  readonly model: Option.Option<ServingModelId>
  readonly reasoningEffort: Option.Option<ResponseReasoningEffort>
}

/** A request as sent: its model chosen. */
export interface ChatRequest {
  readonly model: ServingModelId
  readonly reasoningEffort: Option.Option<ResponseReasoningEffort>
  readonly settings: RequestedResponseSettings
}

export interface ChatDraft {
  readonly text: string
  /** The settings chosen for the next request, over the chat's own. */
  readonly settings: Partial<ChatRequestSettings>
  /** The latest submission: its response and the text it sent. */
  readonly submitted: {
    readonly id: MessageId
    readonly text: string
  } | null
}

export const defaultResponseSettings = Schema.decodeUnknownSync(RequestedResponseSettings)({
  systemPrompt: "You are a helpful assistant chatting inside Magnitude, an app that runs AI models on the user's hardware.",
  stop: [],
})

/** The settings a chat's next request starts from: those of its transcript's latest response, else the defaults. */
export const chatRequestSettings = (messages: ReadonlyArray<ChatMessage>): ChatRequestSettings =>
  Option.match(latestResponse(messages), {
    onNone: () => ({ ...defaultResponseSettings, model: Option.none(), reasoningEffort: Option.none() }),
    onSome: (response) => ({ ...response.settings, model: Option.some(response.model), reasoningEffort: response.reasoningEffort }),
  })

/** The request these settings make, once a model is chosen. */
export const chatRequest = ({ model, reasoningEffort, ...settings }: ChatRequestSettings): Option.Option<ChatRequest> =>
  Option.map(model, (chosen) => ({ model: chosen, reasoningEffort, settings }))

/** Presentation drafts survive navigation without copying the watched chat. */
export const selectedChat = Atom.keepAlive(Atom.make<SessionId | null>(null))
export const chatDraft = Atom.family((id: SessionId | null) =>
  Atom.keepAlive(Atom.make<ChatDraft>({ text: "", settings: {}, submitted: null }))
)

/** Sends text as a new message, with new IDs for the message and its response. */
export const chatSend = (
  chatId: SessionId,
  text: string,
  request: ChatRequest
): Effect.Effect<ChatMessagePayload> =>
  Effect.map(Effect.all([generateMessageId, generateMessageId]), ([messageId, responseId]) => ({
    chatId,
    messageId,
    responseId,
    text,
    ...request,
  }))

/** Requests a new response in place of the latest one. */
export const chatRetry = (
  chatId: SessionId,
  replaces: MessageId,
  request: ChatRequest
): Effect.Effect<ChatRetryPayload> =>
  Effect.map(generateMessageId, (responseId) => ({ chatId, responseId, replaces, ...request }))

export const completeChatSubmission = (
  draft: ChatDraft,
  text: string,
  id: MessageId,
  retry: boolean
): ChatDraft => ({
  ...draft,
  text: !retry && draft.text === text ? "" : draft.text,
  submitted: { id, text },
})

/** A failed response restores its captured input only while the next draft is still empty. */
export const chatComposerText = (draft: ChatDraft, last: ChatAssistantMessage | undefined): string =>
  draft.text ||
  (last !== undefined && responseStatus(last) === "failed" && last.id === draft.submitted?.id ? draft.submitted.text : "")
