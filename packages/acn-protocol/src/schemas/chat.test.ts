import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { ChatMessagePayload, ChatRetryPayload } from "../boundary/chat"
import {
  applyChatUpdate,
  ChatBusy,
  chatMessages,
  ChatNotFound,
  ChatStorageFailed,
  ChatUpdate,
  InvalidChatRequest,
  liveResponse,
  type WatchedChat,
} from "./chat"
import { SessionId } from "./identifier"

const chatId = SessionId.make("chat")

describe("Chat errors", () => {
  it("own their text, which survives the wire", () => {
    const errors = Schema.Union(ChatNotFound, ChatBusy, InvalidChatRequest, ChatStorageFailed)
    for (const [error, message] of [
      [new ChatNotFound({ chatId }), "This conversation no longer exists."],
      [new ChatBusy({ chatId }), "A response is already running in this conversation."],
      [new InvalidChatRequest({ reason: "not_latest_response" }), "Only the last response can be retried."],
      [new ChatStorageFailed({ reason: "unavailable" }), "Chat history could not be read or saved. Check access to the Magnitude data folder."],
    ] as const) {
      const wire = JSON.parse(JSON.stringify(Schema.encodeSync(errors)(error)))
      expect(wire).not.toHaveProperty("message")
      expect(Schema.decodeUnknownSync(errors)(wire).message).toBe(message)
    }
  })
})

const update = Schema.decodeUnknownSync(ChatUpdate)
const assistant = (text: string, ended = false) => ({
  type: "assistant",
  id: "response",
  model: "qwen3-8b",
  settings: { systemPrompt: "", stop: [] },
  started: 0,
  content: text ? [{ type: "text", text, time: { started: 10, ended: 20 } }] : [],
  ...(ended ? { end: { time: 40, outcome: { type: "finished", finishReason: "stop" } } } : {}),
})
const summary = {
  id: "chat",
  title: "Hello",
  time: { created: 0, updated: 0 },
  messageCount: 2,
  running: true,
}
const user = { type: "user", id: "user", text: "Hello", time: 0 }
const snapshot = update({ type: "snapshot", chat: summary, messages: [user, assistant("Hi")] })
const live = (pending?: string) => ({
  id: "response",
  phase: { phase: { type: "generating" } },
  ...(pending === undefined ? {} : { pending: { type: "text", text: pending, time: { started: 25, ended: 30 } } }),
})
const changed = (change: { chat?: unknown; messages?: ReadonlyArray<readonly [number, unknown]>; live?: unknown }) => update({
  type: "changed",
  ...(change.chat === undefined ? {} : { chat: change.chat }),
  messages: (change.messages ?? []).map(([index, message]) => ({ index, message })),
  ...(change.live === undefined ? {} : { live: change.live }),
})
const fold = (...updates: ReadonlyArray<ChatUpdate>) =>
  updates.reduce<Option.Option<WatchedChat>>((previous, next) => Option.some(applyChatUpdate(previous, next)), Option.none())
const open = (watched: Option.Option<WatchedChat>) => {
  const value = Option.getOrThrow(watched)
  if (value._tag !== "open") throw new Error("The chat was removed.")
  return value
}
const textOf = (watched: Extract<WatchedChat, { _tag: "open" }>) => {
  const response = chatMessages(watched)[1]!
  return response._tag === "assistant" ? response.content.map((part) => part.text).join("") : ""
}

describe("Watched chat", () => {
  it("shows a running response's recorded output followed by its pending output", () => {
    const watched = open(fold(snapshot, changed({ live: live(" there") })))
    expect(textOf(watched)).toBe("Hi there")
    const response = watched.messages[1]!
    expect(response._tag === "assistant" && Option.map(Option.flatMap(liveResponse(watched, response), (r) => r.phase), (p) => p.phase._tag))
      .toEqual(Option.some("generating"))
  })

  it("applies a recorded group and the live state it clears in one step, showing the output once", () => {
    const streaming = fold(snapshot, changed({ live: live(" there") }))
    const recorded = open(Option.some(applyChatUpdate(streaming, changed({ messages: [[1, assistant("Hi there")]], live: live() }))))
    expect(textOf(recorded)).toBe("Hi there")
    expect(Option.flatMap(recorded.live, (response) => response.pending)).toEqual(Option.none())
    const ended = open(Option.some(applyChatUpdate(Option.some(recorded), changed({
      chat: { ...summary, running: false },
      messages: [[1, assistant("Hi there", true)]],
      live: null,
    }))))
    expect(textOf(ended)).toBe("Hi there")
    expect(ended.live).toEqual(Option.none())
    expect(ended.chat.running).toBe(false)
  })

  it("sets a message at its index and appends one at the end", () => {
    const appended = open(fold(snapshot, changed({ messages: [[2, { type: "user", id: "next", text: "More", time: 50 }]] })))
    expect(appended.messages.map((m) => m.id)).toEqual(["user", "response", "next"])
    const replaced = open(fold(snapshot, changed({ messages: [[1, assistant("Retried")]] })))
    expect(replaced.messages).toHaveLength(2)
    expect(textOf(replaced)).toBe("Retried")
  })

  it("keeps every part a change leaves out", () => {
    const renamed = open(fold(snapshot, changed({ live: live(" there") }), changed({ chat: { ...summary, title: "Renamed" } })))
    expect(renamed.chat.title).toBe("Renamed")
    expect(renamed.messages).toHaveLength(2)
    expect(textOf(renamed)).toBe("Hi there")
  })

  it("ends in removal", () => {
    expect(Option.getOrThrow(fold(snapshot, changed({ live: live("") }), update({ type: "removed" })))._tag).toBe("removed")
  })

  it("treats an update out of order as a defect: before its snapshot, after removal, or past the end", () => {
    const removed = fold(snapshot, update({ type: "removed" }))
    expect(() => applyChatUpdate(Option.none(), changed({}))).toThrow("began without its snapshot")
    expect(() => applyChatUpdate(Option.none(), update({ type: "removed" }))).toThrow("began without its snapshot")
    expect(() => applyChatUpdate(removed, changed({}))).toThrow("after the chat's removal")
    expect(() => applyChatUpdate(removed, snapshot)).toThrow("after the chat's removal")
    expect(() => fold(snapshot, changed({ messages: [[3, user]] }))).toThrow("set message 3 of 2")
  })
})

describe("Chat payloads", () => {
  const response = { model: "qwen3-8b", settings: { systemPrompt: "", stop: [] } }
  const decodes = <A, I>(schema: Schema.Schema<A, I>, input: unknown) => Option.isSome(Schema.decodeUnknownOption(schema)(input))

  it("refuse one ID naming two messages", () => {
    expect(decodes(ChatMessagePayload, { chatId: "chat", messageId: "a", responseId: "b", text: "Hi", ...response })).toBe(true)
    expect(decodes(ChatMessagePayload, { chatId: "chat", messageId: "a", responseId: "a", text: "Hi", ...response })).toBe(false)
    expect(decodes(ChatRetryPayload, { chatId: "chat", replaces: "a", responseId: "b", ...response })).toBe(true)
    expect(decodes(ChatRetryPayload, { chatId: "chat", replaces: "a", responseId: "a", ...response })).toBe(false)
  })
})
