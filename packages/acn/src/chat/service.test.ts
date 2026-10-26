import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Chunk, DateTime, Deferred, Effect, Fiber, Option, Ref, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { ChatStore } from "@magnitudedev/storage"
import {
  type Change,
  type ChatAssistantMessage,
  type ChatMessage,
  ChatMessagePayload,
  type ChatUpdate,
  type MessageId,
  type ResponseEnded,
  responseMetrics,
  type SessionEvent,
  type SessionId,
} from "@magnitudedev/acn-protocol"
import * as Native from "@magnitudedev/icn-protocol/schemas"
import { InferenceActivity } from "../inference-activity"
import {
  done,
  type Generate,
  measured,
  memoryChatStore,
  model,
  newChat,
  newId,
  output,
  payload,
  progress,
  sent,
  served,
  serviceFixture,
  settings,
  sqliteServiceFixture,
  started,
  text,
  thought,
} from "./chat.test-support"
import { InferenceFailed } from "./inference"
import { ChatService } from "./service"

const fixture = (generate: Generate, store?: Parameters<typeof serviceFixture>[1]) =>
  Effect.runPromise(serviceFixture(generate, store))

/** Folds a watch's updates into the chat it describes. */
const fold = (messages: ReadonlyArray<ChatMessage>, update: ChatUpdate): ReadonlyArray<ChatMessage> => {
  switch (update._tag) {
    case "snapshot":
      return update.messages
    case "changed":
      return update.messages.reduce((current, { index, message }) =>
        index < current.length ? current.map((existing, at) => at === index ? message : existing) : [...current, message], messages)
    case "removed":
      return messages
  }
}

/** The transcript once the response (else the latest one) has ended. */
const ended = (service: ChatService, chatId: SessionId, responseId?: MessageId) =>
  service.watch(chatId).pipe(
    Stream.scan([] as ReadonlyArray<ChatMessage>, fold),
    Stream.filter((messages) => {
      const response = responseId === undefined
        ? messages.at(-1)
        : messages.find((message) => message.id === responseId)
      return response?._tag === "assistant" && Option.isSome(response.end)
    }),
    Stream.runHead,
    Effect.map(Option.getOrThrow),
    Effect.timeout("5 seconds")
  )

const lastResponse = (messages: ReadonlyArray<ChatMessage>): ChatAssistantMessage => {
  const last = messages.at(-1)
  if (last?._tag !== "assistant") throw new Error("Expected a response")
  return last
}

const listChanges = (changes: ReadonlyArray<Change>) => changes.filter((change) => change.operation === "ListChats").length

const endsOf = (events: ReadonlyArray<SessionEvent>) =>
  events.filter((event): event is ResponseEnded => event._tag === "response_ended")

/**
 * Collects a watch in the background, once it has its snapshot, through its removal. A watch never
 * ends on its own, so the watcher stops at `removed`; `released` says whether the watch's own scope
 * has closed since.
 */
const watching = (service: ChatService, chatId: SessionId) => Effect.gen(function* () {
  const subscribed = yield* Deferred.make<void>()
  const released = yield* Deferred.make<void>()
  const fiber = yield* Effect.fork(Stream.runCollect(service.watch(chatId).pipe(
    Stream.ensuring(Deferred.succeed(released, undefined)),
    Stream.tap(() => Deferred.succeed(subscribed, undefined)),
    Stream.takeUntil((update) => update._tag === "removed")
  )))
  yield* Deferred.await(subscribed)
  return {
    updates: Fiber.join(fiber).pipe(Effect.map(Chunk.toReadonlyArray), Effect.timeout("5 seconds")),
    released: Deferred.isDone(released),
  }
})

/** A response that streams one delta, then runs until it is stopped. */
const endless: Generate = () => Stream.concat(Stream.make(text("Partial")), Stream.never)

/** The live state an update shows, if it shows any. */
const liveOf = (update: ChatUpdate) =>
  update._tag === "snapshot" ? update.live : update._tag === "changed" ? Option.flatten(update.live) : Option.none()

/** Waits until a watcher sees the running response's first output. */
const streaming = (service: ChatService, chatId: SessionId) =>
  service.watch(chatId).pipe(
    Stream.filter((update) => Option.exists(liveOf(update), (live) => Option.isSome(live.pending))),
    Stream.runHead,
    Effect.timeout("5 seconds")
  )

describe("Chat generation", () => {
  it("streams a response into output groups and its end, with its request and serving usage", async () => {
    const f = await fixture(() => Stream.fromIterable([
      progress({ phase: "queued" }),
      progress({ phase: "generating" }),
      thought("Consider "),
      thought("it."),
      text("Hel"),
      text("lo"),
      done(),
    ]))
    const chatId = newChat()
    const input = payload(chatId, "Say hello")
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(input)
      const response = lastResponse(yield* ended(service, chatId))
      expect(response.content.map((part) => [part._tag, part.text])).toEqual([["reasoning", "Consider it."], ["text", "Hello"]])
      expect(yield* (yield* InferenceActivity).count).toBe(0)
    }))

    const events = Option.getOrThrow(await Effect.runPromise(f.memory.events(chatId)))
    // Phases are live state only.
    expect(events.map((event) => event._tag)).toEqual([
      "message_sent", "response_started", "response_output", "response_output", "response_ended",
    ])
    const [end] = endsOf(events)
    expect(end?.outcome).toEqual({ _tag: "finished", finishReason: "stop" })
    // Both come from the engine's last timings report: the prompt's tokens include the cached ones.
    expect(end?.usage).toEqual(Option.some({
      promptTokens: 40, cachedTokens: 12, completionTokens: 3, draftTokens: Option.some(8), draftAcceptedTokens: Option.some(6),
    }))
    expect(end?.timings).toEqual(Option.some({ promptMs: 10, predictedMs: 50, samplerMs: 2, parserMs: 1, timeToFirstTokenMs: 15 }))
    expect(Schema.encodeSync(Native.ChatCompletionRequest)(f.requests[0]!)).toEqual({
      model,
      stream: true,
      timings_per_token: true,
      messages: [{ role: "system", content: "Be brief." }, { role: "user", content: "Say hello" }],
      reasoning_effort: "medium",
      temperature: 0.7,
      max_tokens: 256,
      seed: 7,
      stop: ["</end>"],
    })
    expect(f.usage).toEqual([expect.objectContaining({
      id: `chat:${input.responseId}`, model, input: 40, cached: 12, output: 3, generationMs: 50, complete: true,
    })])
  })

  it("sends earlier text back without reasoning or empty responses, and retries in place of the latest response", async () => {
    const replies = [[thought("Hidden"), text("First answer"), done()], [thought("Only thinking"), done()], [text("Retried"), done()]]
    let call = 0
    const f = await fixture(() => Stream.fromIterable(replies[call++]!))
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId, "One"))
      yield* ended(service, chatId)
      const second = payload(chatId, "Two")
      yield* service.send(second)
      yield* ended(service, chatId, second.responseId)
      const retry = newId()
      yield* service.retry({ chatId, responseId: retry, replaces: second.responseId, model, reasoningEffort: Option.none(), settings })
      const messages = yield* ended(service, chatId, retry)
      expect(messages.map((message) => message._tag === "user" ? message.text : message.id))
        .toEqual(["One", expect.any(String), "Two", retry])
      const [chat] = (yield* service.list(0, 10)).chats
      expect(chat?.messageCount).toBe(4)
    }))
    const conversation = f.requests.map((request) =>
      request.messages.map((message) => [message.role, "content" in message ? message.content : undefined]))
    expect(conversation[1]).toEqual([["system", "Be brief."], ["user", "One"], ["assistant", Option.some("First answer")], ["user", "Two"]])
    expect(conversation[2]).toEqual(conversation[1])
  })

  it("records an open group once its second has passed, even when no further chunk arrives", async () => {
    const gate = Effect.runSync(Deferred.make<void>())
    const f = await fixture(() => Stream.concat(
      Stream.make(text("a")),
      Stream.fromEffect(Deferred.await(gate)).pipe(Stream.flatMap(() => Stream.make(text("b"), done())))
    ))
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId))
      const recorded = yield* f.memory.events(chatId).pipe(
        Effect.map(Option.getOrThrow),
        Effect.repeat({ until: (events) => events.some((event) => event._tag === "response_output") }),
        Effect.timeout("3 seconds")
      )
      expect(endsOf(recorded)).toEqual([])
      yield* Deferred.succeed(gate, undefined)
      yield* ended(service, chatId)
    }))
    const groups = Option.getOrThrow(await Effect.runPromise(f.memory.events(chatId)))
      .flatMap((event) => event._tag === "response_output" ? [event.deltas] : [])
    expect(groups).toEqual([["a"], ["b"]])
  })
})

describe("Chat outcomes", () => {
  it("keeps partial output and the engine's error when inference fails", async () => {
    const f = await fixture(() => Stream.concat(
      Stream.make(text("Part")),
      Stream.fail(new InferenceFailed({
        error: { _tag: "inference", code: "server_error", message: "The engine stopped.", param: Option.none() },
      }))
    ))
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId))
      const response = lastResponse(yield* ended(service, chatId))
      expect(response.content.map((part) => part.text)).toEqual(["Part"])
      expect(Option.map(response.end, (end) => end.outcome)).toEqual(Option.some({
        _tag: "failed",
        error: { _tag: "inference", code: "server_error", message: "The engine stopped.", param: Option.none() },
      }))
    }))
  })

  it("records a finish without the engine's timings report as a protocol failure, keeping its output", async () => {
    const unreported = Schema.decodeUnknownSync(Native.ChatCompletionChunk)({
      id: "chatcmpl-test", created: 0, model: "qwen-test-q4", object: "chat.completion.chunk",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    })
    const f = await fixture(() => Stream.make(text("Hi"), unreported))
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId))
      const response = lastResponse(yield* ended(service, chatId))
      expect(response.content.map((part) => part.text)).toEqual(["Hi"])
      expect(Option.map(response.end, (end) => end.outcome)).toEqual(Option.some({
        _tag: "failed",
        error: { _tag: "protocol", message: "The inference stream finished without its timings report." },
      }))
      // Neither is recorded without the report.
      expect(Option.map(response.end, (end) => [end.usage, end.timings])).toEqual(Option.some([Option.none(), Option.none()]))
    }))
  })

  it("records the finish reason verbatim, and a stream that ends unfinished as an unreachable service", async () => {
    const replies = [[text("x"), done("tool_calls")], [text("y")]]
    let call = 0
    const f = await fixture(() => Stream.fromIterable(replies[call++]!))
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId))
      const first = lastResponse(yield* ended(service, chatId))
      expect(Option.map(first.end, (end) => end.outcome)).toEqual(Option.some({ _tag: "finished", finishReason: "tool_calls" }))
      const second = payload(chatId)
      yield* service.send(second)
      const unfinished = lastResponse(yield* ended(service, chatId, second.responseId))
      expect(Option.map(unfinished.end, (end) => end.outcome._tag === "failed" && end.outcome.error._tag)).toEqual(Option.some("unreachable"))
    }))
  })

  it("stops the named response, keeping its output and measurements, and ignores a replayed stop", async () => {
    const f = await fixture(() => Stream.concat(Stream.make(measured(), text("Partial")), Stream.never))
    const chatId = newChat()
    const input = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(input)
      yield* streaming(service, chatId)
      yield* service.stop({ chatId, responseId: input.responseId })
      const response = lastResponse(yield* ended(service, chatId))
      expect(Option.map(response.end, (end) => end.outcome)).toEqual(Option.some({ _tag: "stopped", by: "user" }))
      expect(response.content.map((part) => part.text)).toEqual(["Partial"])
      // The engine's latest timings report counts the tokens a stopped response generated.
      expect(responseMetrics(response).outputTokens).toEqual(Option.some(3))
      yield* service.stop({ chatId, responseId: input.responseId })
      const unknown = yield* Effect.flip(service.stop({ chatId, responseId: newId() }))
      expect(unknown._tag).toBe("InvalidChatRequest")
    }))
    expect(endsOf(Option.getOrThrow(await Effect.runPromise(f.memory.events(chatId))))).toHaveLength(1)
  })

  it("ends a response that could not be recorded when the session next opens", async () => {
    const f = await fixture(endless)
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId))
      yield* streaming(service, chatId)
      // The store stops accepting output: the instance fails, and its watchers are told.
      const watcher = yield* Effect.fork(Stream.runDrain(service.watch(chatId)))
      yield* Ref.set(f.memory.failing, (commit) => Option.isSome(commit))
      const failure = yield* Fiber.join(watcher).pipe(Effect.flip, Effect.timeout("5 seconds"))
      expect(failure).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
    }))
    await Effect.runPromise(Ref.set(f.memory.failing, () => false))
    const restarted = await fixture(endless, f.memory)
    await restarted.run(Effect.gen(function* () {
      const service = yield* ChatService
      const response = lastResponse(yield* ended(service, chatId))
      expect(Option.map(response.end, (end) => end.outcome)).toEqual(Option.some({ _tag: "stopped", by: "shutdown" }))
    }))
  })
})

describe("Chat commands", () => {
  it("returns a replayed send or retry without effect, even when its model is no longer served", async () => {
    const f = await fixture(() => Stream.make(text("ok"), done()))
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId))
      yield* ended(service, chatId)
      const next = payload(chatId)
      yield* service.send(next)
      const messages = yield* ended(service, chatId, next.responseId)
      const retry = { chatId, responseId: newId(), replaces: next.responseId, model, reasoningEffort: Option.none(), settings }
      yield* service.retry(retry)
      yield* ended(service, chatId, retry.responseId)
      yield* Ref.set(f.models, served([]))
      yield* service.send(next)
      yield* service.retry(retry)
      expect(messages).toHaveLength(4)
    }))
    expect(f.requests).toHaveLength(3)
  })

  it("starts only new chats, sends only to existing ones, and leaves nothing for a start it rejects", async () => {
    const f = await fixture(() => Stream.make(text("ok"), done()))
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const unknown = yield* Effect.flip(service.send(payload(newChat())))
      expect(unknown._tag).toBe("ChatNotFound")
      // An unknown chat is not found, whatever its model.
      const unservedUnknown = { ...payload(newChat()), model: "image-test" as typeof model }
      expect((yield* Effect.flip(service.send(unservedUnknown)))._tag).toBe("ChatNotFound")
      expect((yield* Effect.flip(service.retry({ ...unservedUnknown, replaces: newId() })))._tag).toBe("ChatNotFound")
      const unservable = yield* Effect.flip(service.start({ ...payload(chatId), model: "image-test" as typeof model }))
      expect(unservable).toMatchObject({ _tag: "InvalidChatRequest", reason: "model_unavailable" })
      expect((yield* service.list(0, 10)).chats).toEqual([])
      yield* service.start(payload(chatId))
      const reasoning = yield* Effect.flip(service.send({ ...payload(chatId), reasoningEffort: Option.some("extreme") }))
      expect(reasoning).toMatchObject({ _tag: "InvalidChatRequest", reason: "reasoning_unsupported" })
    }))
  })

  it("does nothing for a replayed start, even once its model is not served, and rejects another start of the chat", async () => {
    const f = await fixture(() => Stream.make(text("ok"), done()))
    const chatId = newChat()
    const first = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(first)
      yield* ended(service, chatId)
      const before = Option.getOrThrow(yield* f.memory.events(chatId))
      yield* service.start(first)
      yield* service.start({ ...first, model: "image-test" as typeof model })
      expect(Option.getOrThrow(yield* f.memory.events(chatId))).toEqual(before)
      expect(yield* Effect.flip(service.start(payload(chatId)))).toMatchObject({ _tag: "InvalidChatRequest", reason: "ids_in_use" })
      expect(Option.getOrThrow(yield* f.memory.events(chatId))).toEqual(before)
    }))
    expect(f.requests).toHaveLength(1)
  })

  it("rejects a send while a response runs, and a retry of anything but the latest response", async () => {
    const f = await fixture(endless)
    const chatId = newChat()
    const first = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(first)
      expect((yield* Effect.flip(service.send(payload(chatId))))._tag).toBe("ChatBusy")
      yield* service.stop({ chatId, responseId: first.responseId })
      yield* ended(service, chatId)
      const stale = yield* Effect.flip(service.retry({ chatId, responseId: newId(), replaces: newId(), model, reasoningEffort: Option.none(), settings }))
      expect(stale).toMatchObject({ _tag: "InvalidChatRequest", reason: "not_latest_response" })
    }))
  })

  it("rejects blank text at the payload", () => {
    expect(Schema.decodeUnknownEither(ChatMessagePayload)({ ...Schema.encodeSync(ChatMessagePayload)(payload(newChat())), text: "  " })._tag)
      .toBe("Left")
  })

  it("renames from the title the client saw, takes a replay as done, refuses a stale one, and lists chats by latest activity", async () => {
    const f = await fixture(() => Stream.make(text("ok"), done()))
    const [older, newer] = [newChat(), newChat()]
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(older, "Older"))
      yield* ended(service, older)
      yield* service.start(payload(newer, "Newer"))
      yield* ended(service, newer)
      expect((yield* service.list(0, 10)).chats.map((chat) => chat.title)).toEqual(["Newer", "Older"])
      // The rename is the latest activity.
      yield* Effect.sleep("2 millis")
      yield* service.rename({ chatId: older, from: "Older", title: "Renamed" })
      yield* service.rename({ chatId: older, from: "Older", title: "Renamed" })
      const stale = yield* Effect.flip(service.rename({ chatId: older, from: "Older", title: "Other" }))
      expect(stale).toMatchObject({ _tag: "InvalidChatRequest", reason: "title_changed" })
      const page = yield* service.list(0, 1)
      expect([page.chats.map((chat) => chat.title), page.hasMore]).toEqual([["Renamed"], true])
      expect((yield* service.list(1, 1)).hasMore).toBe(false)
      expect((yield* Effect.flip(service.rename({ chatId: newChat(), from: "", title: "x" })))._tag).toBe("ChatNotFound")
    }))
    // A rename changes the session's record, never its log.
    expect(Option.map(f.memory.record(older), (record) => record.title)).toEqual(Option.some("Renamed"))
    expect(Option.map(await Effect.runPromise(f.memory.events(older)), (events) => events.map((event) => event._tag))).toEqual(Option.some([
      "message_sent", "response_started", "response_output", "response_ended",
    ]))
  })

  it("tells the chat list of each command's summary change: a start, a rename of a chat that isn't open, the first send after reopening", async () => {
    /** Runs a command, then waits until the chat list is told of a change. */
    const tells = <E>(changes: ReadonlyArray<Change>, command: Effect.Effect<void, E>) => Effect.gen(function* () {
      const before = listChanges(changes)
      yield* command
      yield* Effect.sleep("5 millis").pipe(
        Effect.repeat({ until: () => listChanges(changes) > before }),
        Effect.timeout("3 seconds")
      )
    })
    const f = await fixture(endless)
    const chatId = newChat()
    const first = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      // Told when the chat starts, not only once its first response ends.
      yield* tells(f.changes, service.start(first))
      yield* service.stop({ chatId, responseId: first.responseId })
      yield* ended(service, chatId)
    }))
    const renamed = await fixture(endless, f.memory)
    await renamed.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* tells(renamed.changes, service.rename({ chatId, from: "Hi", title: "Renamed" }))
    }))
    const reopened = await fixture(endless, f.memory)
    await reopened.run(Effect.gen(function* () {
      const service = yield* ChatService
      const next = payload(chatId)
      yield* tells(reopened.changes, service.send(next))
      yield* service.stop({ chatId, responseId: next.responseId })
    }))
  })

  it("tells the chat list when an instance fails or a delete fails, and lists the response either left as not running", async () => {
    /** Waits until the chat list is told of a change since `before`. */
    const told = (changes: ReadonlyArray<Change>, before: number) =>
      Effect.sleep("5 millis").pipe(Effect.repeat({ until: () => listChanges(changes) > before }), Effect.timeout("3 seconds"))
    const listedAs = (service: ChatService, chatId: SessionId) =>
      Effect.map(service.list(0, 10), ({ chats }) => chats.find((chat) => chat.id === chatId)?.running)

    const failing = await fixture(endless)
    const failed = newChat()
    await failing.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(failed))
      yield* streaming(service, failed)
      const watcher = yield* Effect.fork(Stream.runDrain(service.watch(failed)))
      const before = listChanges(failing.changes)
      // The store stops accepting commits: the instance fails with its response still running.
      yield* Ref.set(failing.memory.failing, (commit) => Option.isSome(commit))
      yield* Effect.flip(Fiber.join(watcher)).pipe(Effect.timeout("5 seconds"))
      yield* told(failing.changes, before)
      yield* Ref.set(failing.memory.failing, () => false)
      expect(yield* listedAs(service, failed)).toBe(false)
    }))

    const deleting = await fixture(endless)
    const kept = newChat()
    await deleting.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(kept))
      yield* streaming(service, kept)
      const before = listChanges(deleting.changes)
      yield* Ref.set(deleting.memory.failing, () => true)
      expect(yield* Effect.flip(service.remove(kept))).toMatchObject({ _tag: "ChatStorageFailed" })
      expect(listChanges(deleting.changes)).toBeGreaterThan(before)
      // The removal closed its instance, which recorded no end: with no instance open, it is not running.
      yield* Ref.set(deleting.memory.failing, () => false)
      expect(yield* listedAs(service, kept)).toBe(false)
    }))
  })

  it("removes a chat with nothing left behind, also while it runs, and tells its watchers", async () => {
    const f = await fixture(() => Stream.concat(Stream.make(measured(), text("Partial")), Stream.never))
    const chatId = newChat()
    const input = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(input)
      yield* streaming(service, chatId)
      const watched = yield* watching(service, chatId)
      yield* service.remove(chatId)
      const updates = yield* watched.updates
      expect(updates.at(-1)).toEqual({ _tag: "removed" })
      // The watcher stopping at removal releases the watch, and with it the removed chat's instance.
      expect(yield* watched.released).toBe(true)
      // The response it cut short is still recorded as served, with what was observed.
      expect(f.usage).toEqual([expect.objectContaining({ id: `chat:${input.responseId}`, output: 3, generationMs: 50, complete: false })])
      expect(yield* f.memory.events(chatId)).toEqual(Option.none())
      expect((yield* service.list(0, 10)).chats).toEqual([])
      expect((yield* Effect.flip(Stream.runDrain(service.watch(chatId))))._tag).toBe("ChatNotFound")
      yield* service.remove(chatId)
    }))
  })

  it("fails a removed chat's watch with the store's reason when its delete fails, and reopens the chat on its next use", async () => {
    const f = await fixture(endless)
    const chatId = newChat()
    const input = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(input)
      yield* streaming(service, chatId)
      const watched = yield* watching(service, chatId)
      yield* Ref.set(f.memory.failing, () => true)
      expect(yield* Effect.flip(service.remove(chatId))).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      // The watcher is never told `removed` of a chat that still exists.
      expect(yield* Effect.flip(watched.updates)).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      yield* Ref.set(f.memory.failing, () => false)
      const reopened = yield* service.watch(chatId).pipe(Stream.runHead, Effect.map(Option.getOrThrow), Effect.timeout("5 seconds"))
      expect(reopened).toMatchObject({ _tag: "snapshot", chat: { id: chatId, running: false } })
      // The response the removal cut short ended when the chat reopened.
      expect(Option.map(lastResponse(reopened._tag === "snapshot" ? reopened.messages : []).end, (end) => end.outcome))
        .toEqual(Option.some({ _tag: "stopped", by: "shutdown" }))
      yield* service.send(payload(chatId))
    }))
  })

  it("completes a removal whose caller is interrupted, and tells its watchers only once the delete has succeeded", async () => {
    const f = await fixture(endless)
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId))
      yield* streaming(service, chatId)
      const reached = yield* Deferred.make<void>()
      const gate = yield* Deferred.make<void>()
      yield* Ref.set(f.memory.deleting, Effect.zipRight(Deferred.succeed(reached, undefined), Deferred.await(gate)))
      const watched = yield* watching(service, chatId)
      const watcher = yield* Effect.fork(watched.updates)
      const removal = yield* Effect.fork(service.remove(chatId))
      // The instance has closed as removed, and the delete waits.
      yield* Deferred.await(reached)
      const interrupting = yield* Effect.fork(Fiber.interrupt(removal))
      yield* Effect.sleep("20 millis")
      expect(yield* Fiber.poll(watcher)).toEqual(Option.none())
      expect(yield* Fiber.poll(interrupting)).toEqual(Option.none())
      yield* Deferred.succeed(gate, undefined)
      yield* Fiber.join(interrupting)
      expect((yield* Fiber.join(watcher)).at(-1)).toEqual({ _tag: "removed" })
      expect(yield* f.memory.events(chatId)).toEqual(Option.none())
      expect((yield* service.list(0, 10)).chats).toEqual([])
    }))
  })

  it("fails only chat calls when the store fails", async () => {
    const memory = await Effect.runPromise(memoryChatStore)
    await Effect.runPromise(Ref.set(memory.failing, () => true))
    const f = await fixture(endless, memory)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      expect(yield* Effect.flip(service.list(0, 10))).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      expect(yield* Effect.flip(service.start(payload(newChat())))).toMatchObject({ _tag: "ChatStorageFailed" })
      // Once the store recovers, so do chats.
      yield* Ref.set(memory.failing, () => false)
      expect((yield* service.list(0, 10)).chats).toEqual([])
    }))
  })
})

describe("Chat recovery", () => {
  const seedRunning = Effect.gen(function* () {
    const memory = yield* memoryChatStore
    const chatId = newChat()
    const response = newId()
    yield* memory.seed(chatId, [sent(newId(), "Hi", 1_000), started(response, 1_000),
      output(response, "text", 1_100, ["Par", "tial"], [0, 400]),
    ])
    return { memory, chatId }
  })

  it("lists a response a crash left running as not running, and ends it when its session next opens, at the last moment it was observed", async () => {
    const { memory, chatId } = await Effect.runPromise(seedRunning)
    const f = await fixture(endless, memory)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const [chat] = (yield* service.list(0, 10)).chats
      expect(chat).toMatchObject({ id: chatId, running: false, messageCount: 2 })
      // Listing opens no session, so nothing is recorded yet.
      expect(endsOf(Option.getOrThrow(yield* memory.events(chatId)))).toEqual([])
      const response = lastResponse(yield* ended(service, chatId))
      expect(Option.map(response.end, (end) => end.outcome)).toEqual(Option.some({ _tag: "stopped", by: "shutdown" }))
    }))
    const [end] = endsOf(Option.getOrThrow(await Effect.runPromise(memory.events(chatId))))
    expect(end?.outcome).toEqual({ _tag: "stopped", by: "shutdown" })
    expect(DateTime.toEpochMillis(end!.time)).toBe(1_500)
    // The end is the session's latest observation.
    expect(Option.map(memory.record(chatId), (record) => DateTime.toEpochMillis(record.time.updated))).toEqual(Option.some(1_500))
    expect(f.requests).toEqual([])
  })

  it("fails only a session's own calls when its log does not decode, and still removes it", async () => {
    const { memory, chatId } = await Effect.runPromise(seedRunning)
    memory.corrupt.add(chatId)
    const f = await fixture(() => Stream.make(text("ok"), done()), memory)
    const healthy = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(healthy))
      yield* ended(service, healthy)
      expect((yield* service.list(0, 10)).chats.map((chat) => chat.id).sort()).toEqual([chatId, healthy].sort())
      expect(yield* Effect.flip(Stream.runDrain(service.watch(chatId)))).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      expect(yield* Effect.flip(service.send(payload(chatId)))).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      yield* service.rename({ chatId: healthy, from: "Hi", title: "Still here" })
      yield* service.remove(chatId)
      expect((yield* service.list(0, 10)).chats.map((chat) => chat.title)).toEqual(["Still here"])
      expect(yield* memory.events(chatId)).toEqual(Option.none())
    }))
  })

  it("fails the list, and the session's own calls, while its record does not decode", async () => {
    const { memory, chatId } = await Effect.runPromise(seedRunning)
    memory.unreadable.add(chatId)
    const f = await fixture(endless, memory)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      expect(yield* Effect.flip(service.list(0, 10))).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      expect(yield* Effect.flip(service.send(payload(chatId)))).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      expect(yield* Effect.flip(Stream.runDrain(service.watch(chatId)))).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      memory.unreadable.delete(chatId)
      expect((yield* service.list(0, 10)).chats.map((chat) => chat.id)).toEqual([chatId])
    }))
  })
})

describe("Chat record", () => {
  it("counts in the record exactly the messages each transcript shows, through sends, retries, stops and failures", async () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-chat-record-"))
    try {
      // Each response finishes, fails, or runs until it is stopped, in turn.
      const replies: ReadonlyArray<ReadonlyArray<Native.ChatCompletionChunk> | "fail" | "endless"> = [
        [text("One"), done()], "fail", "endless", [text("Two"), done()], "endless", [text("Three"), done()],
      ]
      let call = 0
      const f = await Effect.runPromise(sqliteServiceFixture((request) => {
        const reply = replies[call++ % replies.length]!
        return reply === "fail"
          ? Stream.fail(new InferenceFailed({ error: { _tag: "inference", code: "server_error", message: "No.", param: Option.none() } }))
          : reply === "endless" ? endless(request) : Stream.fromIterable(reply)
      }, root))
      const [first, second] = [newChat(), newChat()]
      const counted = (service: ChatService, store: ChatStore, chatId: SessionId) => Effect.gen(function* () {
        const transcript = yield* ended(service, chatId)
        const record = Option.getOrThrow(yield* store.record(chatId))
        const [listed] = (yield* service.list(0, 10)).chats.filter((chat) => chat.id === chatId)
        return [record.messageCount, listed?.messageCount, transcript.length]
      })
      /** Ends the running response, by its own end or by a stop. */
      const settle = (service: ChatService, chatId: SessionId, responseId: MessageId) => Effect.gen(function* () {
        const running = yield* service.watch(chatId).pipe(Stream.runHead, Effect.map((update) =>
          Option.exists(update, (snapshot) => snapshot._tag === "snapshot" && snapshot.chat.running)))
        if (running) yield* service.stop({ chatId, responseId })
      })
      await f.run(Effect.gen(function* () {
        const service = yield* ChatService
        const store = yield* ChatStore
        const counts: Array<ReadonlyArray<number | undefined>> = []
        for (const chatId of [first, second]) {
          const start = payload(chatId)
          yield* service.start(start)
          yield* settle(service, chatId, start.responseId)
          counts.push(yield* counted(service, store, chatId))
          let latest = start.responseId
          for (const step of ["send", "retry", "send", "retry", "send"] as const) {
            const responseId = newId()
            if (step === "send") yield* service.send({ ...payload(chatId), responseId })
            else yield* service.retry({ chatId, responseId, replaces: latest, model, reasoningEffort: Option.none(), settings })
            latest = responseId
            yield* Effect.sleep("5 millis").pipe(Effect.zipRight(settle(service, chatId, responseId)))
            counts.push(yield* counted(service, store, chatId))
          }
        }
        for (const [record, listed, transcript] of counts) {
          expect(record).toBe(transcript)
          expect(listed).toBe(transcript)
        }
        expect(counts.map(([, , transcript]) => transcript)).toEqual([2, 4, 4, 6, 6, 8, 2, 4, 4, 6, 6, 8])
      }))
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  }, 20_000)
})

describe("Chat watch", () => {
  it("sends a snapshot, then one change per settlement it observes, then removal", async () => {
    const gate = Effect.runSync(Deferred.make<void>())
    const f = await fixture(() => Stream.concat(
      Stream.make(progress({ phase: "prefill", completed_tokens: 5, total_tokens: 10, cached_tokens: 2 }), text("Hi")),
      Stream.fromEffect(Deferred.await(gate)).pipe(Stream.flatMap(() => Stream.make(done())))
    ))
    const chatId = newChat()
    const input = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(input)
      yield* streaming(service, chatId)
      const watched = yield* watching(service, chatId)
      yield* Deferred.succeed(gate, undefined)
      yield* ended(service, chatId)
      yield* service.rename({ chatId, from: "Hi", title: "Renamed" })
      yield* service.remove(chatId)
      const updates = yield* watched.updates

      const [first, ...rest] = updates
      if (first?._tag !== "snapshot") throw new Error("A watch begins with its snapshot")
      expect(first.chat.id).toBe(chatId)
      const live = Option.getOrThrow(first.live)
      expect(Option.map(live.pending, (part) => part.text)).toEqual(Option.some("Hi"))
      expect(Option.flatMap(live.phase, (phase) => phase.progress)).toEqual(Option.some({
        _tag: "prefill", completedTokens: 5, totalTokens: 10, cachedTokens: 2,
      }))
      expect(rest.map((update) => update._tag)).toEqual([...rest.slice(0, -1).map(() => "changed"), "removed"])
      const changes = rest.flatMap((update) => update._tag === "changed" ? [update] : [])
      // The settlement that ends the response carries its end, the summary and the cleared live state at once.
      const ending = changes.find((update) => update.messages.some(({ message }) => message._tag === "assistant" && Option.isSome(message.end)))
      expect(ending?.live).toEqual(Option.some(Option.none()))
      expect(Option.map(ending!.chat, (chat) => chat.running)).toEqual(Option.some(false))
      // Every change that records the group clears the live output in the same update.
      for (const change of changes.filter((update) => update.messages.some(({ message }) => message._tag === "assistant" && message.content.length > 0))) {
        expect(Option.exists(change.live, (live) => !Option.exists(live, (response) => Option.isSome(response.pending)))).toBe(true)
      }
      expect(changes.some((update) => Option.exists(update.chat, (chat) => chat.title === "Renamed"))).toBe(true)
      // Folding the updates gives the transcript as recorded, its output shown once.
      const messages = updates.reduce(fold, [])
      expect(lastResponse(messages).content.map((part) => part.text)).toEqual(["Hi"])
    }))
  })

  it("recreates a removed chat from a start replayed after the removal, with exactly that first message", async () => {
    const f = await fixture(() => Stream.make(text("ok"), done()))
    const chatId = newChat()
    const first = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(first)
      yield* ended(service, chatId)
      yield* service.remove(chatId)
      yield* service.start(first)
      yield* ended(service, chatId)
      expect((yield* service.list(0, 10)).chats.map((chat) => chat.id)).toEqual([chatId])
    }))
    const events = Option.getOrThrow(await Effect.runPromise(f.memory.events(chatId)))
    expect(events.slice(0, 2)).toMatchObject([
      { _tag: "message_sent", messageId: first.messageId, text: first.text },
      { _tag: "response_started", responseId: first.responseId },
    ])
    expect(events.filter((event) => event._tag === "message_sent" || event._tag === "response_started")).toHaveLength(2)
  })

  it("fails for a chat that does not exist", async () => {
    const f = await fixture(endless)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      expect((yield* Effect.flip(Stream.runDrain(service.watch(newChat()))))._tag).toBe("ChatNotFound")
    }))
  })
})
