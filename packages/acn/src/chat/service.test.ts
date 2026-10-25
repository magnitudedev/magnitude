import { BunContext } from "@effect/platform-bun"
import { Chunk, Deferred, Effect, Fiber, Layer, Option, Schema, Stream } from "effect"
import { Database } from "bun:sqlite"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import {
  ChatCommandId,
  ChatConversationId,
  ChatFailed,
  ChatModelId,
  defaultChatSettings,
  type ChatGeneration,
  type ChatSettings,
} from "@magnitudedev/acn-protocol"
import {
  ChatStorage,
  ChatStorageLive,
  GlobalStorage,
  makeGlobalStorage,
} from "@magnitudedev/storage"
import { BunSqliteDriverLayer } from "@magnitudedev/storage/sqlite/bun"
import * as Native from "@magnitudedev/icn-protocol/schemas"
import { AcnChangesLive } from "../changes"
import { InferenceActivityLive, InferenceActivity } from "../inference-activity"
import { ServingUsage, type ServingUsageRecord } from "../serving-usage"
import { ChatInference } from "./inference"
import { ChatService, ChatServiceLive } from "./service"
const roots: string[] = []
afterEach(() => {
  for (const path of roots.splice(0))
    rmSync(path, { recursive: true, force: true })
})
const root = () => {
  const path = mkdtempSync(join(tmpdir(), "magnitude-chat-"))
  roots.push(path)
  return path
}
const model = {
  id: "qwen-test",
  name: "Qwen test",
  description: "",
  created: 0,
  object: "model",
  owned_by: "local",
  context_length: 32768,
  architecture: { input_modalities: ["text"], output_modalities: ["text"] },
  supported_parameters: ["reasoning"],
  top_provider: { context_length: 32768, max_completion_tokens: 32768 },
  reasoning: {
    default_effort: "medium",
    default_enabled: true,
    mandatory: false,
    supported_efforts: ["off", "low", "medium", "high"],
  },
}
const models = Schema.decodeUnknownSync(Native.OpenAiModelsResponse)({
  object: "list",
  data: [model],
})
const settings: ChatSettings = {
  ...defaultChatSettings,
  modelId: Option.some(ChatModelId.make(model.id)),
  reasoningEffort: Option.some("medium"),
}
const chunk = (patch: Record<string, unknown>) =>
  Schema.decodeUnknownSync(Native.ChatCompletionChunk)({
    id: "test",
    created: 0,
    model: model.id,
    object: "chat.completion.chunk",
    choices: [],
    ...patch,
  })
const text = (s: string) =>
  chunk({ choices: [{ index: 0, delta: { content: s } }] })
const thought = chunk({
  choices: [{ index: 0, delta: { reasoning_content: "Consider the setup." } }],
})
const done = (finish = "stop") =>
  chunk({
    choices: [{ index: 0, delta: {}, finish_reason: finish }],
    usage: {
      prompt_tokens: 40,
      completion_tokens: 3,
      total_tokens: 43,
      prompt_tokens_details: { cached_tokens: 12 },
    },
    timings: {
      cache_n: 12,
      parser_ms: 0,
      predicted_ms: 50,
      predicted_n: 3,
      predicted_per_second: 60,
      predicted_per_token_ms: 16.67,
      prompt_ms: 10,
      prompt_n: 40,
      prompt_per_second: 4000,
      prompt_per_token_ms: 0.25,
      sampler_ms: 0,
      time_to_first_token_ms: 15,
    },
  })
function fixture(
  path = root(),
  generate: ChatInference["generate"] = () =>
    Stream.fromIterable([thought, text("Hello"), done()]),
  writable = () => true
) {
  const requests: Native.ChatCompletionRequest[] = [],
    usage: ServingUsageRecord[] = []
  const baseStorage = ChatStorageLive.pipe(
    Layer.provideMerge(
      Layer.mergeAll(
        BunContext.layer,
        BunSqliteDriverLayer,
        Layer.succeed(GlobalStorage, makeGlobalStorage({ root: path }))
      )
    )
  )
  const storage = Layer.effect(
    ChatStorage,
    Effect.map(ChatStorage, (service) => ({
      ...service,
      save: (generation) =>
        writable()
          ? service.save(generation)
          : Effect.fail(
              new ChatFailed({ code: "Storage", message: "Disk is full" })
            ),
    }))
  ).pipe(Layer.provideMerge(baseStorage))
  const deps = Layer.mergeAll(
    storage,
    AcnChangesLive,
    InferenceActivityLive,
    Layer.succeed(ServingUsage, {
      record: (r) =>
        Effect.sync(() => {
          usage.push(r)
        }),
      read: () => Effect.die("Unused"),
    }),
    Layer.succeed(ChatInference, {
      models: Effect.succeed(models),
      generate: (r) => {
        requests.push(r)
        return generate(r)
      },
    })
  )
  const layer = ChatServiceLive.pipe(Layer.provideMerge(deps))
  const run = <A, E>(
    effect: Effect.Effect<A, E, ChatService | ChatStorage | InferenceActivity>
  ) => Effect.runPromise(Effect.scoped(effect.pipe(Effect.provide(layer))))
  return { run, requests, usage, path }
}
const id = () => ChatConversationId.make(crypto.randomUUID())
const command = () => ChatCommandId.make(crypto.randomUUID())
const create = (s: ChatService) => s.create(id(), settings)
const start = (
  s: ChatService,
  c: ChatConversationId,
  content = "Hi",
  custom = settings,
  cmd = command()
) =>
  s.start({
    conversationId: c,
    commandId: cmd,
    content: Option.some(content),
    retryMessageId: Option.none(),
    settings: custom,
  })
const terminal = (s: ChatService, c: ChatConversationId, previous?: string) =>
  s.watch(c).pipe(
    Stream.filterMap((u) =>
      u._tag === "Generation"
        ? Option.some(u.generation).pipe(Option.filter(g => g.status !== "running" && g.id !== previous))
        : u._tag === "Snapshot"
        ? Option.fromNullable(
            u.conversation.messages.at(-1)?.generation.pipe(
              Option.filter((g) => g.status !== "running" && g.id !== previous),
              Option.getOrUndefined
            )
          )
        : Option.none<ChatGeneration>()
    ),
    Stream.runHead,
    Effect.map(Option.getOrThrow),
    Effect.timeout("3 seconds")
  )

describe("durable Chat", () => {
  it("streams text, reasoning and telemetry, stores exact context and counts usage once", async () => {
    const f = fixture()
    const g = await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const c = yield* create(s)
        yield* start(s, c.id, "Explain prefill")
        const g = yield* terminal(s, c.id)
        expect(yield* (yield* InferenceActivity).count).toBe(0)
        return g
      })
    )
    expect(g.content).toBe("Hello")
    expect(g.reasoning).toBe("Consider the setup.")
    expect(g.status).toBe("completed")
    expect(g.metrics.outputTokens).toEqual(Option.some(3))
    expect(g.metrics.tokensPerSecond).toEqual(Option.some(60))
    expect(Option.isSome(g.metrics.ttftMs)).toBe(true)
    expect(f.usage).toHaveLength(1)
    expect(f.requests).toHaveLength(1)
    const request = Schema.encodeSync(Native.ChatCompletionRequest)(
      f.requests[0]!
    )
    expect(request.tools).toBeUndefined()
    const systemPrompt = request.messages[0]?.content as string
    expect(systemPrompt).toBe(defaultChatSettings.systemPrompt)
    expect(systemPrompt).not.toContain("Current setup")
    expect(systemPrompt).not.toContain("inferenceHost")
    expect(systemPrompt).not.toContain("App layout")
    expect(systemPrompt).not.toContain("Selected model:")
    const db = new Database(join(f.path, "chats.sqlite"), { readonly: true })
    expect(
      JSON.stringify(db.query("SELECT data FROM generations").all())
    ).toContain("infoVersion")
    db.close()
  })
  it("deduplicates request IDs, rejects overlapping sends and acknowledges durable Stop", async () => {
    const f = fixture(undefined, () =>
      Stream.concat(Stream.succeed(text("Partial")), Stream.never)
    )
    await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const c = yield* create(s)
        const cmd = command()
        const g = yield* start(s, c.id, "First", settings, cmd)
        expect(yield* start(s, c.id, "First", settings, cmd)).toBe(g)
        const busy = yield* start(s, c.id).pipe(Effect.either)
        expect(busy._tag === "Left" && busy.left.code).toBe("Busy")
        yield* s.stop(c.id, g)
        expect((yield* terminal(s, c.id)).status).toBe("stopped")
        expect(yield* (yield* InferenceActivity).count).toBe(0)
        yield* s.stop(c.id, g)
      })
    )
    expect(f.requests).toHaveLength(1)
    expect(f.usage).toHaveLength(1)
  })
  it("survives viewer disconnect and returns a current snapshot on reconnect", async () => {
    const f = fixture(undefined, () =>
      Stream.concat(
        Stream.succeed(text("Hello")),
        Stream.fromEffect(Effect.sleep("150 millis").pipe(Effect.as(done())))
      )
    )
    await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const c = yield* create(s)
        yield* start(s, c.id)
        yield* s.watch(c.id).pipe(Stream.take(1), Stream.runDrain)
        const g = yield* terminal(s, c.id)
        expect(g.status).toBe("completed")
        expect(g.content).toBe("Hello")
      })
    )
    expect(f.requests).toHaveLength(1)
  })
  it("preserves partial output and an actionable engine failure", async () => {
    const f = fixture(undefined, () =>
      Stream.concat(
        Stream.succeed(text("Partial answer")),
        Stream.fail(
          new ChatFailed({
            code: "Inference",
            message: "Context window exceeded",
          })
        )
      )
    )
    await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const c = yield* create(s)
        yield* start(s, c.id)
        const g = yield* terminal(s, c.id)
        expect(g.status).toBe("failed")
        expect(g.content).toBe("Partial answer")
        expect(g.error).toEqual(Option.some("Context window exceeded"))
      })
    )
    expect(f.usage[0]?.complete).toBe(false)
  })
  it("distinguishes unfinished streams and output limits", async () => {
    for (const [stream, status] of [
      [Stream.succeed(text("Partial")), "interrupted"],
      [Stream.fromIterable([text("Short"), done("length")]), "length"],
    ] as const) {
      const f = fixture(undefined, () => stream)
      await f.run(
        Effect.gen(function* () {
          const s = yield* ChatService
          const c = yield* create(s)
          yield* start(s, c.id)
          expect((yield* terminal(s, c.id)).status).toBe(status)
        })
      )
    }
  })
  it("retries the last response with current settings and keeps previous attempts", async () => {
    const f = fixture()
    const customPrompt = " \nAnswer in French.\n "
    await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const c = yield* create(s)
        yield* start(s, c.id, "Question")
        const first = yield* terminal(s, c.id)
        yield* s.start({
          conversationId: c.id,
          commandId: command(),
          content: Option.none(),
          retryMessageId: Option.some(first.messageId),
          settings: {
            ...settings,
            systemPrompt: customPrompt,
            temperature: Option.some(0.2),
            reasoningEffort: Option.some("high"),
          },
        })
        yield* terminal(s, c.id, first.id)
        expect((yield* s.get(c.id)).messages).toHaveLength(2)
        yield* start(s, c.id, "Next question")
        yield* terminal(s, c.id)
        const bad = yield* s
          .start({
            conversationId: c.id,
            commandId: command(),
            content: Option.none(),
            retryMessageId: Option.some(first.messageId),
            settings,
          })
          .pipe(Effect.either)
        expect(bad._tag === "Left" && bad.left.code).toBe("InvalidRequest")
      })
    )
    expect(f.requests[1]?.messages).toHaveLength(2)
    expect(f.requests[1]?.temperature).toEqual(Option.some(0.2))
    const retryRequest = Schema.encodeSync(Native.ChatCompletionRequest)(
      f.requests[1]!
    )
    expect(retryRequest.messages[0]?.content).toBe(customPrompt)
    const db = new Database(join(f.path, "chats.sqlite"), { readonly: true })
    expect(db.query("SELECT COUNT(*) n FROM generations").get()).toEqual({
      n: 3,
    })
    db.close()
  })
  it("retains chronological history across twenty turns and long prompts", async () => {
    const f = fixture()
    await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const c = yield* create(s)
        let previous: string | undefined
        for (let i = 0; i < 20; i++) {
          yield* start(s, c.id, `Turn ${i}: ${"Long prompt. ".repeat(i * 10)}`)
          previous = (yield* terminal(s, c.id, previous)).id
        }
        expect((yield* s.get(c.id)).messages).toHaveLength(40)
      })
    )
    expect(f.requests.at(-1)?.messages).toHaveLength(40)
    expect(f.requests.at(-1)?.messages[1]?.content).toEqual("Turn 0: ")
    expect(f.usage).toHaveLength(20)
  })
  it("rejects empty text, unsupported effort and unavailable models before persistence", async () => {
    const f = fixture()
    await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const c = yield* create(s)
        for (const [text, custom] of [
          ["  ", settings],
          ["Hi", { ...settings, reasoningEffort: Option.some("super") }],
          [
            "Hi",
            { ...settings, modelId: Option.some(ChatModelId.make("missing")) },
          ],
        ] as const) {
          const r = yield* start(s, c.id, text, custom).pipe(Effect.either)
          expect(r._tag === "Left" && r.left.code).toBe("InvalidRequest")
        }
        expect((yield* s.get(c.id)).messages).toHaveLength(0)
      })
    )
    expect(f.requests).toHaveLength(0)
  })
  it("paginates and deletes chats including active responses with cascading records", async () => {
    const f = fixture(undefined, () => Stream.never)
    await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const chats = yield* Effect.forEach(Array.from({ length: 28 }), () =>
          create(s)
        )
        expect((yield* s.list(0, 25)).hasMore).toBe(true)
        expect((yield* s.list(25, 25)).conversations).toHaveLength(3)
        const c = chats[0]!
        yield* s.update(c.id, Option.some("Renamed"), Option.none())
        expect((yield* s.get(c.id)).title).toBe("Renamed")
        yield* start(s, c.id)
        yield* s.remove(c.id)
        expect((yield* s.get(c.id).pipe(Effect.either))._tag).toBe("Left")
      })
    )
    const db = new Database(join(f.path, "chats.sqlite"), { readonly: true })
    expect(db.query("SELECT COUNT(*) n FROM generations").get()).toEqual({
      n: 0,
    })
    expect(db.query("SELECT COUNT(*) n FROM messages").get()).toEqual({ n: 0 })
    db.close()
  })
  it("reopens SQLite and recovers abandoned generations after a process restart", async () => {
    const f = fixture()
    let conversationId = id()
    await f.run(
      Effect.gen(function* () {
        const s = yield* ChatService
        const c = yield* s.create(conversationId, settings)
        yield* start(s, c.id)
        yield* terminal(s, c.id)
      })
    )
    const db = new Database(join(f.path, "chats.sqlite"))
    const row = db.query("SELECT id,data FROM generations").get() as {
      id: string
      data: string
    }
    const g = JSON.parse(row.data)
    g.status = "running"
    db.query("UPDATE generations SET status='running',data=? WHERE id=?").run(
      JSON.stringify(g),
      row.id
    )
    db.close()
    const second = fixture(f.path)
    await second.run(
      Effect.gen(function* () {
        const c = yield* (yield* ChatService).get(conversationId)
        expect(c.messages).toHaveLength(2)
        expect(
          c.messages[1]?.generation.pipe(Option.map((g) => g.status))
        ).toEqual(Option.some("interrupted"))
      })
    )
    expect(second.requests).toHaveLength(0)
  })
  it("preserves unreadable or newer history instead of overwriting it", async () => {
    const path = root()
    writeFileSync(join(path, "chats.sqlite"), "not a SQLite database")
    await fixture(path).run(
      Effect.gen(function* () {
        const r = yield* (yield* ChatService).list(0, 25).pipe(Effect.either)
        expect(r._tag === "Left" && r.left.code).toBe("Storage")
      })
    )
    const newer = root()
    const db = new Database(join(newer, "chats.sqlite"))
    db.exec("PRAGMA user_version=99")
    db.close()
    await fixture(newer).run(
      Effect.gen(function* () {
        const r = yield* (yield* ChatService).list(0, 25).pipe(Effect.either)
        expect(r._tag === "Left" && r.left.message).toContain("newer")
      })
    )
  })
  it("keeps settings changes out of an in-flight request", async () => {
    const f = fixture(undefined, () =>
      Stream.concat(
        Stream.succeed(text("First")),
        Stream.fromEffect(Effect.sleep("100 millis").pipe(Effect.as(done())))
      )
    )
    await f.run(
      Effect.gen(function* () {
        const service = yield* ChatService
        const chat = yield* create(service)
        yield* start(service, chat.id)
        yield* service.update(
          chat.id,
          Option.none(),
          Option.some({
            ...settings,
            temperature: Option.some(1.7),
            systemPrompt: "Answer in French.",
          })
        )
        const response = yield* terminal(service, chat.id)
        expect(response.settings.systemPrompt).toBe(
          defaultChatSettings.systemPrompt
        )
        expect((yield* service.get(chat.id)).settings.systemPrompt).toBe(
          "Answer in French."
        )
      })
    )
    expect(f.requests[0]?.temperature).toEqual(Option.none())
  })
  it("closes active requests durably when the service scope closes", async () => {
    const f = fixture(undefined, () => Stream.never)
    let chatId = id()
    await f.run(
      Effect.gen(function* () {
        const service = yield* ChatService
        yield* service.create(chatId, settings)
        yield* start(service, chatId)
      })
    )
    await fixture(f.path).run(
      Effect.gen(function* () {
        const service = yield* ChatService
        const c = yield* service.get(chatId)
        expect(
          c.messages.at(-1)?.generation.pipe(Option.map((g) => g.status))
        ).toEqual(Option.some("interrupted"))
      })
    )
  })
  it("ignores role-only and empty chunks when measuring the first token", async () => {
    const f = fixture(undefined, () =>
      Stream.concat(
        Stream.fromIterable([
          chunk({
            choices: [{ index: 0, delta: { role: "assistant", content: "" } }],
          }),
          chunk({ progress: { phase: "preparing" } }),
        ]),
        Stream.concat(
          Stream.fromEffect(
            Effect.sleep("100 millis").pipe(Effect.as(thought))
          ),
          Stream.fromIterable([text("Answer"), done()])
        )
      )
    )
    await f.run(
      Effect.gen(function* () {
        const service = yield* ChatService
        const c = yield* create(service)
        yield* start(service, c.id)
        const g = yield* terminal(service, c.id)
        expect(Option.getOrThrow(g.metrics.ttftMs)).toBeGreaterThanOrEqual(90)
        expect(
          Option.getOrThrow(g.metrics.firstAnswerMs)
        ).toBeGreaterThanOrEqual(Option.getOrThrow(g.metrics.ttftMs))
      })
    )
  })
  it("commits progress checkpoints before completion and resumes viewing without replaying requests", async () => {
    const f = fixture(undefined, () =>
      Stream.concat(
        Stream.succeed(text("First")),
        Stream.concat(
          Stream.fromEffect(
            Effect.sleep("1050 millis").pipe(Effect.as(text(" second")))
          ),
          Stream.never
        )
      )
    )
    await f.run(
      Effect.gen(function* () {
        const service = yield* ChatService
        const c = yield* create(service)
        const generation = yield* start(service, c.id)
        yield* service.watch(c.id).pipe(
          Stream.filter(
            (u) =>
              u._tag === "Generation" ? u.generation.content === "First second" : u._tag === "Snapshot" &&
              u.conversation.messages.at(-1)?.content === "First second"
          ),
          Stream.take(1),
          Stream.runDrain,
          Effect.timeout("2 seconds")
        )
        const db = new Database(join(f.path, "chats.sqlite"), {
          readonly: true,
        })
        const row = db
          .query("SELECT data FROM generations WHERE id=?")
          .get(generation) as { data: string }
        expect(JSON.parse(row.data).content).toBe("First second")
        db.close()
        yield* service.stop(c.id, generation)
      })
    )
    expect(f.requests).toHaveLength(1)
  })

  it("keeps a response visible when its terminal save fails, records usage and retries the write before another send", async () => {
    let writable = false
    const f = fixture(undefined, undefined, () => writable)
    await f.run(
      Effect.gen(function* () {
        const service = yield* ChatService
        const chat = yield* create(service)
        yield* start(service, chat.id)
        const first = yield* terminal(service, chat.id)
        expect(first.content).toBe("Hello")
        expect(Option.getOrThrow(first.error)).toContain("could not be saved")
        const blocked = yield* start(service, chat.id, "Next").pipe(
          Effect.either
        )
        expect(blocked._tag === "Left" && blocked.left.code).toBe("Storage")
        expect((yield* service.get(chat.id)).messages).toHaveLength(2)
        writable = true
        yield* start(service, chat.id, "Next")
        yield* terminal(service, chat.id, first.id)
        expect((yield* service.get(chat.id)).messages).toHaveLength(4)
      })
    )
    expect(f.usage).toHaveLength(2)
  })
  it.each([
    [{ phase: "queued" }, "queued"],
    [{ phase: "preparing" }, "preparing"],
    [{ phase: "model_loading", stage: "optimizing", fraction: .2 }, "optimizing"],
    [{ phase: "model_loading", stage: "loading_weights", fraction: .6 }, "loading"],
    [{ phase: "model_loading", stage: "finalizing", fraction: .9 }, "finalizing"],
    [{ phase: "prefill", completed_tokens: 10, total_tokens: 20, cached_tokens: 5 }, "prefill"],
  ])("flushes a paused %j status without waiting for another engine event", async (progress, phase) => {
    const f = fixture(undefined, () => Stream.concat(Stream.succeed(chunk({ progress })), Stream.never))
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const chat = yield* create(service)
      const generationId = yield* start(service, chat.id)
      yield* service.watch(chat.id).pipe(Stream.filter(update => {
        const generation = update._tag === "Generation" ? update.generation : update._tag === "Snapshot" ? update.conversation.messages.at(-1)?.generation.pipe(Option.getOrUndefined) : undefined
        return !!generation && Option.exists(generation.activity, a => a.phase === phase)
      }), Stream.take(1), Stream.runDrain, Effect.timeout("1 second"))
      yield* service.stop(chat.id, generationId)
      expect((yield* terminal(service, chat.id)).status).toBe("stopped")
    }))
  })
  it("coalesces slow-viewer output without losing admitted messages or terminal state", async () => {
    const f = fixture()
    await f.run(Effect.gen(function* () {
      const service = yield* ChatService
      const chat = yield* create(service)
      const ready = yield* Deferred.make<void>()
      const watched = yield* Effect.fork(service.watch(chat.id).pipe(
        Stream.tap(() => Deferred.succeed(ready, undefined).pipe(Effect.zipRight(Effect.sleep("100 millis")))),
        Stream.takeUntil(update => update._tag === "Snapshot" && update.conversation.messages.some(m => Option.exists(m.generation, g => g.status !== "running"))),
        Stream.runCollect,
      ))
      yield* Deferred.await(ready)
      yield* start(service, chat.id)
      const updates = yield* Fiber.join(watched).pipe(Effect.timeout("1 second"))
      const last = Chunk.last(updates).pipe(Option.getOrThrow)
      expect(last._tag).toBe("Snapshot")
      if (last._tag === "Snapshot") { expect(last.conversation.messages).toHaveLength(2); expect(last.conversation.messages[1]?.content).toBe("Hello") }
    }))
  })
  it("does not repeat a completed command after rebuilding the service", async () => {
    const f = fixture()
    const chatId = id(), cmd = command()
    let generationId: string
    await f.run(Effect.gen(function* () { const service = yield* ChatService; yield* service.create(chatId, settings); generationId = yield* start(service, chatId, "Original", settings, cmd); yield* terminal(service, chatId) }))
    const second = fixture(f.path)
    await second.run(Effect.gen(function* () { const service = yield* ChatService; expect(yield* start(service, chatId, "Original", settings, cmd)).toBe(generationId); expect((yield* service.get(chatId)).messages).toHaveLength(2) }))
    expect(second.requests).toHaveLength(0)
  })

})
