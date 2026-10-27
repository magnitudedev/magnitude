import { Chunk, DateTime, Deferred, Effect, Either, Exit, Fiber, Layer, Option, Queue, Ref, Stream, Take, TestClock, TestContext, type Scope } from "effect"
import { describe, expect, it } from "vitest"
import {
  applyChatUpdate,
  type Change,
  type ChatAssistantMessage,
  type ChatMessage,
  chatMessages,
  type ChatUpdate,
  type MessageId,
  type ResponseEnded,
  type SessionEvent,
  type SessionId,
  type WatchedChat,
} from "@magnitudedev/acn-protocol"
import { ChatStore } from "@magnitudedev/storage"
import type * as Native from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges } from "../changes"
import { InferenceActivity, InferenceActivityLive } from "../inference-activity"
import { ServingUsage, type ServingUsageRecord } from "../serving-usage"
import {
  done,
  type MemoryChatStore,
  memoryChatStore,
  model,
  newChat,
  newId,
  finished,
  output,
  payload,
  sent,
  served,
  servedModel,
  settings,
  started,
  text,
  thought,
} from "./chat.test-support"
import { ChatInference, InferenceFailed } from "./inference"
import { ChatService, ChatServiceLive } from "./service"

// Inference the test drives: each generate call hands the test a queue of takes to feed it.
type Feed = Queue.Queue<Take.Take<Native.ChatCompletionChunk, InferenceFailed>>

const makeFixture = (options: {
  readonly memory?: MemoryChatStore
  readonly record?: (record: ServingUsageRecord) => Effect.Effect<void>
} = {}) => Effect.gen(function* () {
  const memory = options.memory ?? (yield* memoryChatStore)
  const feeds = yield* Queue.unbounded<Feed>()
  const finalized = yield* Ref.make(0)
  const requests = yield* Ref.make(0)
  const usage: ServingUsageRecord[] = []
  const changes: Change[] = []
  const generate = () => Stream.unwrapScoped(Effect.gen(function* () {
    yield* Ref.update(requests, (n) => n + 1)
    const feed: Feed = yield* Queue.unbounded<Take.Take<Native.ChatCompletionChunk, InferenceFailed>>()
    yield* Queue.offer(feeds, feed)
    return Stream.fromQueue(feed).pipe(Stream.flattenTake)
  })).pipe(Stream.ensuring(Ref.update(finalized, (n) => n + 1)))
  const deps = Layer.mergeAll(
    Layer.succeed(ChatStore, memory.store),
    Layer.succeed(AcnChanges, { publish: (change) => Effect.sync(() => { changes.push(change) }), stream: Stream.empty }),
    InferenceActivityLive,
    Layer.succeed(ServingUsage, {
      record: options.record ?? ((record) => Effect.sync(() => { usage.push(record) })),
      read: () => Effect.die("Unused"),
    }),
    Layer.succeed(ChatInference, { models: Effect.succeed(served([servedModel])), generate }),
  )
  const layer = ChatServiceLive.pipe(Layer.provideMerge(deps))
  const run = <A, E>(effect: Effect.Effect<A, E, ChatService | InferenceActivity | Scope.Scope>) =>
    Effect.runPromise(Effect.scoped(effect.pipe(Effect.provide(layer))))
  /** The next generate call's feed. */
  const next = Queue.take(feeds).pipe(Effect.timeout("3 seconds"))
  return { run, memory, feeds, next, finalized: Ref.get(finalized), requests: Ref.get(requests), usage, changes }
})
const fixture = (options?: Parameters<typeof makeFixture>[0]) => Effect.runPromise(makeFixture(options))

const feed = (q: Feed, ...chunks: Native.ChatCompletionChunk[]) => Effect.forEach(chunks, (chunk) => Queue.offer(q, Take.of(chunk)), { discard: true })
const finish = (q: Feed) => Effect.zipRight(feed(q, done()), Queue.offer(q, Take.end))
const failWith = (q: Feed, code: string) =>
  Queue.offer(q, Take.fail(new InferenceFailed({ error: { _tag: "inference", code, message: `${code} failure`, param: Option.none() } })))

const realSleep = (ms: number) => Effect.promise(() => new Promise<void>((resolve) => setTimeout(resolve, ms)))

const eventsOf = (memory: MemoryChatStore, chatId: SessionId) =>
  Effect.map(memory.events(chatId), Option.getOrElse((): ReadonlyArray<SessionEvent> => []))
const endsOf = (events: ReadonlyArray<SessionEvent>) => events.filter((e): e is ResponseEnded => e._tag === "response_ended")

/** Polls the store until the response has ended. */
const endOf = (memory: MemoryChatStore, chatId: SessionId, responseId: MessageId) =>
  Effect.gen(function* () {
    for (let i = 0; i < 300; i += 1) {
      const end = endsOf(yield* eventsOf(memory, chatId)).find((e) => e.responseId === responseId)
      if (end !== undefined) return end
      yield* realSleep(10)
    }
    return yield* Effect.dieMessage(`response ${responseId} never ended`)
  })

/** Whether an update ends what a watcher follows: a watch never ends on its own, so a watcher stops at removal. */
const isRemoved = (update: ChatUpdate) => update._tag === "removed"

/**
 * Collects a watch in the background, once it has its snapshot, through its removal; `released`
 * says whether the watch's own scope has closed since.
 */
const watching = (service: ChatService, chatId: SessionId) => Effect.gen(function* () {
  const subscribed = yield* Deferred.make<void>()
  const released = yield* Deferred.make<void>()
  const fiber = yield* Effect.fork(Stream.runCollect(service.watch(chatId).pipe(
    Stream.ensuring(Deferred.succeed(released, undefined)),
    Stream.tap(() => Deferred.succeed(subscribed, undefined)),
    Stream.takeUntil(isRemoved)
  )))
  yield* Deferred.await(subscribed).pipe(Effect.timeout("3 seconds"))
  return {
    updates: Fiber.join(fiber).pipe(Effect.map(Chunk.toReadonlyArray), Effect.timeout("5 seconds")),
    released: Deferred.isDone(released),
  }
})

const snapshotOf = (service: ChatService, chatId: SessionId) =>
  service.watch(chatId).pipe(Stream.runHead, Effect.map(Option.getOrThrow), Effect.timeout("3 seconds"))

const lastResponse = (messages: ReadonlyArray<ChatMessage>): ChatAssistantMessage => {
  const last = messages.at(-1)
  if (last?._tag !== "assistant") throw new Error("Expected a response")
  return last
}

/** Starts a chat and finishes its first response. */
const startFinished = (service: ChatService, f: Effect.Effect.Success<ReturnType<typeof makeFixture>>, chatId: SessionId) =>
  Effect.gen(function* () {
    const first = payload(chatId)
    yield* service.start(first)
    yield* finish(yield* f.next)
    yield* endOf(f.memory, chatId, first.responseId)
    return first
  })

describe("adversarial: concurrent commands", () => {
  it("two different sends racing on one chat: exactly one wins, the other is ChatBusy", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* startFinished(service, f, chatId)
      for (let round = 0; round < 10; round += 1) {
        const [a, b] = [payload(chatId, `a${round}`), payload(chatId, `b${round}`)]
        const results = yield* Effect.all([Effect.either(service.send(a)), Effect.either(service.send(b))], { concurrency: "unbounded" })
        const winners = results.filter(Either.isRight)
        expect(winners).toHaveLength(1)
        const loser = results.find(Either.isLeft)
        expect(loser && Either.isLeft(loser) && loser.left._tag).toBe("ChatBusy")
        const won = Either.isRight(results[0]) ? a : b
        yield* finish(yield* f.next)
        yield* endOf(f.memory, chatId, won.responseId)
      }
      expect(yield* f.requests).toBe(11)
    }))
  })

  it("the same send replayed concurrently commits once and both calls succeed", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* startFinished(service, f, chatId)
      const again = payload(chatId)
      const results = yield* Effect.all(Array.from({ length: 8 }, () => Effect.either(service.send(again))), { concurrency: "unbounded" })
      expect(results.every(Either.isRight)).toBe(true)
      const events = yield* eventsOf(f.memory, chatId)
      expect(events.filter((e) => e._tag === "message_sent" && e.messageId === again.messageId)).toHaveLength(1)
      expect(events.filter((e) => e._tag === "response_started" && e.responseId === again.responseId)).toHaveLength(1)
      yield* finish(yield* f.next)
      yield* endOf(f.memory, chatId, again.responseId)
    }))
  })

  it("a send racing a stop: the stopped response ends stopped/user, and every started response ends once", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      let running = yield* startFinished(service, f, chatId)
      running = payload(chatId)
      yield* service.send(running)
      let current = yield* f.next
      for (let round = 0; round < 8; round += 1) {
        yield* feed(current, text(`r${round}`))
        const next = payload(chatId)
        const [, sendResult] = yield* Effect.all([
          service.stop({ chatId, responseId: running.responseId }),
          Effect.either(service.send(next)),
        ], { concurrency: "unbounded" })
        const end = yield* endOf(f.memory, chatId, running.responseId)
        expect(end.outcome).toEqual({ _tag: "stopped", by: "user" })
        if (Either.isLeft(sendResult)) {
          expect(sendResult.left._tag).toBe("ChatBusy")
          yield* service.send(next)
        }
        running = next
        current = yield* f.next
      }
      yield* finish(current)
      yield* endOf(f.memory, chatId, running.responseId)
      const events = yield* eventsOf(f.memory, chatId)
      const startedIds = events.flatMap((e) => e._tag === "response_started" ? [e.responseId] : [])
      const endedIds = endsOf(events).map((e) => e.responseId)
      expect(endedIds).toEqual(startedIds)
    }))
  })

  it("stopping a response queued behind one that is still finishing ends it stopped/user without generating, and spares the finishing one", async () => {
    const recording = Effect.runSync(Deferred.make<void>())
    const releaseRecord = Effect.runSync(Deferred.make<void>())
    const recorded: string[] = []
    let block = false
    const f = await fixture({
      record: (record) => Effect.gen(function* () {
        if (block) {
          yield* Deferred.succeed(recording, undefined)
          yield* Deferred.await(releaseRecord)
        }
        recorded.push(record.id)
      }),
    })
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* startFinished(service, f, chatId)
      const second = payload(chatId)
      yield* service.send(second)
      block = true
      yield* finish(yield* f.next)
      // The second response has ended; its handler is recording usage.
      yield* Deferred.await(recording)
      const third = payload(chatId)
      yield* service.send(third)
      yield* service.stop({ chatId, responseId: third.responseId })
      yield* Deferred.succeed(releaseRecord, undefined)
      const end = yield* endOf(f.memory, chatId, third.responseId)
      expect(end.outcome).toEqual({ _tag: "stopped", by: "user" })
      expect(yield* f.requests).toBe(2)
      expect(recorded).toContain(`chat:${second.responseId}`)
      expect(endsOf(yield* eventsOf(f.memory, chatId)).find((e) => e.responseId === second.responseId)?.outcome)
        .toEqual({ _tag: "finished", finishReason: "stop" })
    }))
  })

  it("stopping an ended response does not stop the one now running", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = yield* startFinished(service, f, chatId)
      const second = payload(chatId)
      yield* service.send(second)
      const q = yield* f.next
      yield* service.stop({ chatId, responseId: first.responseId })
      yield* realSleep(20)
      yield* finish(q)
      expect((yield* endOf(f.memory, chatId, second.responseId)).outcome).toEqual({ _tag: "finished", finishReason: "stop" })
    }))
  })

  it("a StartChat replayed after success does nothing and duplicates nothing", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      yield* service.start(first)
      yield* finish(yield* f.next)
      yield* endOf(f.memory, chatId, first.responseId)
      const events = yield* eventsOf(f.memory, chatId)
      expect(events.map((e) => e._tag)).toEqual(["message_sent", "response_started", "response_ended"])
      expect(yield* f.requests).toBe(1)
    }))
  })
})

describe("adversarial: removal", () => {
  it("removes a streaming chat: watchers get removed, nothing is written after, no instance or stream is left", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(payload(chatId))
      const q = yield* f.next
      yield* feed(q, thought("thinking"), text("partial"))
      const watched = yield* watching(service, chatId)
      yield* realSleep(20)
      yield* service.remove(chatId)
      // Output the stream would still have produced goes nowhere.
      yield* feed(q, text("more"))
      const updates = yield* watched.updates
      expect(updates.at(-1)).toEqual({ _tag: "removed" })
      // The watcher stopping at removal releases the watch, and with it the removed chat's instance.
      expect(yield* watched.released).toBe(true)
      yield* realSleep(30)
      expect(yield* f.memory.events(chatId)).toEqual(Option.none())
      expect(yield* f.finalized).toBe(1)
      expect(yield* (yield* InferenceActivity).count).toBe(0)
      expect((yield* service.list(0, 10)).chats).toEqual([])
      // The service is not stuck: another chat runs.
      const other = newChat()
      const p = payload(other)
      yield* service.start(p)
      yield* finish(yield* f.next)
      yield* endOf(f.memory, other, p.responseId)
    }))
  })

  it("a send replayed after the chat was removed is ChatNotFound and creates nothing", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* startFinished(service, f, chatId)
      const second = payload(chatId)
      yield* service.send(second)
      yield* finish(yield* f.next)
      yield* endOf(f.memory, chatId, second.responseId)
      yield* service.remove(chatId)
      expect((yield* Effect.flip(service.send(second)))._tag).toBe("ChatNotFound")
      expect((yield* Effect.flip(service.send(payload(chatId))))._tag).toBe("ChatNotFound")
      expect((yield* service.list(0, 10)).chats).toEqual([])
      expect(yield* eventsOf(f.memory, chatId)).toEqual([])
    }))
  })

  it("remove racing start: each chat ends consistent (listed iff it has events), and never hangs", async () => {
    const f = await fixture()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      // Answer every generate call at once.
      yield* Effect.forkScoped(Effect.forever(Effect.flatMap(Queue.take(f.feeds), finish)))
      const chats = Array.from({ length: 20 }, newChat)
      const results = yield* Effect.forEach(chats, (chatId, i) => Effect.all([
        Effect.either(service.start(payload(chatId))),
        Effect.zipRight(Effect.repeatN(Effect.yieldNow(), i % 4), service.remove(chatId)),
      ], { concurrency: "unbounded" }).pipe(Effect.timeout("3 seconds")), { concurrency: "unbounded" })
      for (const [start] of results) {
        if (Either.isLeft(start)) expect(start.left).toMatchObject({ _tag: "InvalidChatRequest", reason: "ids_in_use" })
      }
      yield* realSleep(100)
      const listed = new Set((yield* service.list(0, 100)).chats.map((chat) => chat.id))
      for (const chatId of chats) {
        const events = yield* eventsOf(f.memory, chatId)
        expect(listed.has(chatId)).toBe(events.length > 0)
        if (events.length > 0) expect(endsOf(events)).toHaveLength(1)
      }
    }))
  })

  it("watch racing remove ends (removed or ChatNotFound) and never hangs", async () => {
    const f = await fixture()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* Effect.forkScoped(Effect.forever(Effect.flatMap(Queue.take(f.feeds), (q) => feed(q, text("x")))))
      for (let round = 0; round < 15; round += 1) {
        const chatId = newChat()
        yield* service.start(payload(chatId))
        const [watched] = yield* Effect.all([
          Effect.exit(Stream.runCollect(service.watch(chatId).pipe(Stream.takeUntil(isRemoved))).pipe(Effect.timeout("3 seconds"))),
          Effect.zipRight(Effect.repeatN(Effect.yieldNow(), round % 5), service.remove(chatId)),
        ], { concurrency: "unbounded" })
        if (Exit.isSuccess(watched)) {
          expect(Chunk.toReadonlyArray(watched.value).at(-1)).toEqual({ _tag: "removed" })
        } else {
          expect(watched.cause).toMatchObject({ _tag: "Fail", error: { _tag: "ChatNotFound" } })
        }
      }
    }))
  })
})

describe("adversarial: failures", () => {
  it("inference failing before the first token ends the response failed with that error", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      yield* failWith(yield* f.next, "server_error")
      const end = yield* endOf(f.memory, chatId, first.responseId)
      expect(end.outcome).toMatchObject({ _tag: "failed", error: { _tag: "inference", code: "server_error" } })
      const snapshot = yield* snapshotOf(service, chatId)
      expect(snapshot._tag === "snapshot" && lastResponse(snapshot.messages).content).toEqual([])
      expect(snapshot._tag === "snapshot" && snapshot.chat.running).toBe(false)
    }))
  })

  it("inference failing mid-stream keeps the output so far, in order, and ends failed", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      const q = yield* f.next
      yield* feed(q, thought("a"), thought("b"), text("c"), text("d"))
      yield* failWith(q, "overloaded")
      const end = yield* endOf(f.memory, chatId, first.responseId)
      expect(end.outcome).toMatchObject({ _tag: "failed", error: { _tag: "inference", code: "overloaded" } })
      const snapshot = yield* snapshotOf(service, chatId)
      if (snapshot._tag !== "snapshot") return yield* Effect.dieMessage("snapshot")
      expect(lastResponse(snapshot.messages).content.map((p) => [p._tag, p.text])).toEqual([["reasoning", "ab"], ["text", "cd"]])
      expect(snapshot.live).toEqual(Option.none())
    }))
  })

  it("output that cannot be recorded ends the response failed/storage, never stuck running", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      const q = yield* f.next
      yield* Ref.set(f.memory.failing, (commit) => Option.exists(commit, (events) => events.some((e) => e._tag === "response_output")))
      // The part change flushes the reasoning group, which the store refuses.
      yield* feed(q, thought("a"), text("b"))
      const end = yield* endOf(f.memory, chatId, first.responseId)
      expect(end.outcome).toMatchObject({ _tag: "failed", error: { _tag: "storage" } })
      yield* Ref.set(f.memory.failing, () => false)
      const [chat] = (yield* service.list(0, 10)).chats
      expect(chat?.running).toBe(false)
      expect(yield* f.finalized).toBe(1)
    }))
  })

  it("a store that fails every write mid-stream fails the instance; the response is recovered on reopen and the list never shows it running", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      const q = yield* f.next
      yield* feed(q, text("partial"))
      yield* realSleep(20)
      const watcher = yield* Effect.fork(Stream.runDrain(service.watch(chatId)))
      yield* realSleep(20)
      yield* Ref.set(f.memory.failing, (commit) => Option.isSome(commit))
      yield* feed(q, thought("switch"))
      expect(yield* Fiber.join(watcher).pipe(Effect.flip, Effect.timeout("3 seconds"))).toMatchObject({ _tag: "ChatStorageFailed" })
      yield* Ref.set(f.memory.failing, () => false)
      // The store is back. The chat list must not show a response that nothing runs.
      const listed = (yield* service.list(0, 10)).chats[0]
      expect(listed?.running).toBe(false)
    }))
  })

  it("a failed instance is reopened by its next use, which ends the running response stopped/shutdown", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      const q = yield* f.next
      yield* feed(q, text("partial"))
      yield* realSleep(20)
      yield* Ref.set(f.memory.failing, (commit) => Option.isSome(commit))
      yield* feed(q, thought("switch"))
      yield* realSleep(50)
      yield* Ref.set(f.memory.failing, () => false)
      const snapshot = yield* snapshotOf(service, chatId)
      if (snapshot._tag !== "snapshot") return yield* Effect.dieMessage("snapshot")
      expect(Option.map(lastResponse(snapshot.messages).end, (end) => end.outcome)).toEqual(Option.some({ _tag: "stopped", by: "shutdown" }))
      expect(snapshot.chat.running).toBe(false)
      expect(yield* f.finalized).toBe(1)
    }))
  })
})

describe("adversarial: idle close", () => {
  it("does not close an instance whose response is still running, however long it runs", async () => {
    const f = await fixture()
    const chatId = newChat()
    const first = payload(chatId)
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* service.start(first)
      const q = yield* f.next
      yield* feed(q, text("slow"))
      yield* realSleep(20)
      yield* TestClock.adjust("10 minutes")
      yield* realSleep(20)
      yield* TestClock.adjust("10 minutes")
      yield* realSleep(20)
      expect(endsOf(yield* eventsOf(f.memory, chatId))).toEqual([])
      yield* feed(q, text(" but done"))
      yield* finish(q)
      for (let i = 0; i < 100 && endsOf(yield* eventsOf(f.memory, chatId)).length === 0; i += 1) yield* realSleep(10)
      const [end] = endsOf(yield* eventsOf(f.memory, chatId))
      expect(end?.outcome).toEqual({ _tag: "finished", finishReason: "stop" })
    }).pipe(Effect.provide(TestContext.TestContext)))
  })
})

describe("adversarial: watch", () => {
  it("replaying every update through applyChatUpdate gives the final transcript; the shown output only grows and is never duplicated", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      const q = yield* f.next
      yield* feed(q, thought("t0"))
      const watched = yield* watching(service, chatId)
      const parts = [thought("t1"), thought("t2"), text("x1"), text("x2"), thought("t3"), text("x3"), text("x4")]
      for (const part of parts) {
        yield* feed(q, part)
        yield* realSleep(5)
      }
      // A group flushed by time, with no further chunk.
      yield* realSleep(1_100)
      yield* feed(q, text("x5"))
      yield* finish(q)
      yield* endOf(f.memory, chatId, first.responseId)
      // A retry replaces the last message.
      const retry = { chatId, responseId: newId(), replaces: first.responseId, model, reasoningEffort: Option.none(), settings }
      yield* service.retry(retry)
      const r = yield* f.next
      for (const part of [text("y1"), thought("u1"), text("y2")]) {
        yield* feed(r, part)
        yield* realSleep(5)
      }
      yield* finish(r)
      yield* endOf(f.memory, chatId, retry.responseId)
      yield* realSleep(20)
      yield* service.rename({ chatId, from: "Hi", title: "Renamed" })
      yield* realSleep(20)
      yield* service.remove(chatId)
      const updates = yield* watched.updates

      // Every step, the response being shown shows a prefix of what it ends with.
      const finals = new Map<MessageId, string>([
        [first.responseId, "t0t1t2x1x2t3x3x4x5"],
        [retry.responseId, "y1u1y2"],
      ])
      let state = Option.none<WatchedChat>()
      let lastShown = new Map<MessageId, string>()
      const violations: string[] = []
      for (const [index, update] of updates.slice(0, -1).entries()) {
        state = Option.some(applyChatUpdate(state, update))
        const context = `after update ${index} (${update._tag}), next ${updates[index + 1]?._tag}`
        const open = Option.getOrThrow(state)
        if (open._tag !== "open") return yield* Effect.dieMessage("removed early")
        const shown = new Map<MessageId, string>()
        for (const message of chatMessages(open)) {
          if (message._tag !== "assistant") continue
          const all = message.content.map((p) => p.text).join("")
          shown.set(message.id, all)
          if (!finals.get(message.id)!.startsWith(all)) violations.push(`${all} is not a prefix ${context}`)
          if (all.length < (lastShown.get(message.id)?.length ?? 0)) violations.push(`${all} shrank ${context}`)
        }
        lastShown = shown
      }
      const final = Option.getOrThrow(state)
      if (final._tag !== "open") return yield* Effect.dieMessage("removed early")
      expect(final.messages).toHaveLength(2)
      expect(lastResponse(final.messages).id).toBe(retry.responseId)
      expect(lastResponse(final.messages).content.map((p) => [p._tag, p.text])).toEqual([["text", "y1"], ["reasoning", "u1"], ["text", "y2"]])
      expect(final.chat.title).toBe("Renamed")
      expect(final.chat.running).toBe(false)
      expect(final.live).toEqual(Option.none())
      expect(applyChatUpdate(state, updates.at(-1)!)).toEqual({ _tag: "removed" })
      if (violations.length > 0) console.log(violations.join("\n"))
      expect(violations).toHaveLength(0)
    }))
  })

  it("at every update, including the one that records a group, the shown output only grows and is never duplicated", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      const q = yield* f.next
      yield* feed(q, thought("t0"))
      const watched = yield* watching(service, chatId)
      for (const part of [thought("t1"), text("x1"), text("x2"), thought("t2"), text("x3")]) {
        yield* feed(q, part)
        yield* realSleep(5)
      }
      yield* finish(q)
      yield* endOf(f.memory, chatId, first.responseId)
      yield* realSleep(20)
      yield* service.remove(chatId)
      const updates = yield* watched.updates
      let state = Option.none<WatchedChat>()
      let checked = 0
      for (const [index, update] of updates.slice(0, -1).entries()) {
        state = Option.some(applyChatUpdate(state, update))
        const open = Option.getOrThrow(state)
        if (open._tag !== "open") return yield* Effect.dieMessage("removed early")
        const all = lastResponse(chatMessages(open)).content.map((p) => p.text).join("")
        expect("t0t1x1x2t2x3".startsWith(all), `${all} after update ${index}`).toBe(true)
        checked += 1
      }
      expect(checked).toBeGreaterThan(3)
      const final = Option.getOrThrow(state)
      expect(final._tag === "open" && lastResponse(final.messages).content.map((p) => p.text)).toEqual(["t0t1", "x1x2", "t2", "x3"])
    }))
  })

  it("a fresh snapshot equals the replayed state of a watcher that followed the whole response", async () => {
    const f = await fixture()
    const chatId = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const first = payload(chatId)
      yield* service.start(first)
      const q = yield* f.next
      yield* feed(q, text("a"))
      const followed = yield* Effect.fork(service.watch(chatId).pipe(
        Stream.scan(Option.none<WatchedChat>(), (state, update: ChatUpdate) => Option.some(applyChatUpdate(state, update))),
        Stream.takeUntil((state) => Option.exists(state, (s) => s._tag === "open" && !s.chat.running && Option.isNone(s.live) && Option.isSome(lastResponse(s.messages).end))),
        Stream.runLast,
      ))
      yield* realSleep(10)
      for (let n = 0; n < 30; n += 1) yield* feed(q, n % 7 === 0 ? thought(`r${n}`) : text(`${n},`))
      yield* finish(q)
      const replayed = Option.getOrThrow(Option.flatten(yield* Fiber.join(followed).pipe(Effect.timeout("5 seconds"))))
      const snapshot = yield* snapshotOf(service, chatId)
      if (snapshot._tag !== "snapshot" || replayed._tag !== "open") return yield* Effect.dieMessage("snapshot")
      expect(replayed.messages).toEqual(snapshot.messages)
      expect(replayed.chat).toEqual(snapshot.chat)
      expect(replayed.live).toEqual(snapshot.live)
    }))
  })
})

describe("adversarial: recovery and listing", () => {
  it("a crashed response is never listed as running, and ends stopped/shutdown when its session opens, at the later of its start and its last output", async () => {
    const memory = await Effect.runPromise(memoryChatStore)
    const [early, bare] = [newChat(), newChat()]
    const [earlyR, bareR] = [newId(), newId()]
    await Effect.runPromise(Effect.all([
      memory.seed(early, [sent(newId(), "Hi", 1_000), started(earlyR, 1_000), output(earlyR, "reasoning", 1_100, ["a", "b"], [0, 700])]),
      memory.seed(bare, [sent(newId(), "Hi", 1_000), started(bareR, 1_000)]),
    ]))
    const f = await fixture({ memory })
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      expect((yield* service.list(0, 10)).chats.map((chat) => chat.running)).toEqual([false, false])
      const time = (chatId: SessionId) => Effect.map(eventsOf(memory, chatId), (events) => endsOf(events).map((e) => [e.outcome, DateTime.toEpochMillis(e.time)]))
      // Nothing is recorded until a session opens.
      expect([yield* time(early), yield* time(bare)]).toEqual([[], []])
      for (const chatId of [early, bare]) yield* snapshotOf(service, chatId)
      expect(yield* time(early)).toEqual([[{ _tag: "stopped", by: "shutdown" }, 1_800]])
      expect(yield* time(bare)).toEqual([[{ _tag: "stopped", by: "shutdown" }, 1_000]])
      expect((yield* service.list(0, 10)).chats.map((chat) => [chat.id, chat.running, DateTime.toEpochMillis(chat.time.updated)]))
        .toEqual([[early, false, 1_800], [bare, false, 1_000]])
      expect(yield* f.requests).toBe(0)
    }))
  })

  it("a session whose log does not decode is isolated, and is not listed as running", async () => {
    const memory = await Effect.runPromise(memoryChatStore)
    const broken = newChat()
    const r = newId()
    await Effect.runPromise(memory.seed(broken, [sent(newId(), "Hi", 1_000), started(r, 1_000)]))
    memory.corrupt.add(broken)
    const f = await fixture({ memory })
    const healthy = newChat()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      yield* startFinished(service, f, healthy)
      expect(yield* Effect.flip(Stream.runDrain(service.watch(broken)))).toMatchObject({ _tag: "ChatStorageFailed", reason: "unreadable" })
      const page = yield* service.list(0, 10)
      expect(page.chats.map((c) => c.id).sort()).toEqual([broken, healthy].sort())
      expect(page.chats.find((c) => c.id === broken)?.running).toBe(false)
    }))
  })

  it("lists and pages 230 chats by latest activity with no duplicates or gaps", async () => {
    const memory = await Effect.runPromise(memoryChatStore)
    const ids = Array.from({ length: 230 }, newChat)
    await Effect.runPromise(Effect.forEach(ids, (chatId, n) => {
      const r = newId()
      return memory.seed(chatId, [sent(newId(), `chat ${n}`, n), started(r, n), finished(r, 10_000 + (n % 50))])
    }, { discard: true }))
    const f = await fixture({ memory })
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const pages = yield* Effect.forEach([0, 100, 200], (offset) => service.list(offset, 100))
      expect(pages.map((p) => [p.chats.length, p.hasMore])).toEqual([[100, true], [100, true], [30, false]])
      const listed = pages.flatMap((p) => p.chats)
      expect(new Set(listed.map((c) => c.id)).size).toBe(230)
      for (let i = 1; i < listed.length; i += 1) {
        expect(DateTime.toEpochMillis(listed[i - 1]!.time.updated)).toBeGreaterThanOrEqual(DateTime.toEpochMillis(listed[i]!.time.updated))
      }
      expect((yield* service.list(230, 100))).toEqual({ chats: [], hasMore: false })
    }))
  })
})
