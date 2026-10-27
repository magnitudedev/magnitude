import { DateTime, Effect, Layer, Option, type ParseResult, Ref, Schema, type Scope, Stream } from "effect"
import { EventStore, type MemoryEventStore } from "@magnitudedev/event-engine"
import {
  type Change,
  ChatSummary,
  generateMessageId,
  generateSessionId,
  type MessageId,
  type MessageSent,
  type RequestedResponseSettings,
  type ResponseSettings,
  type ResponseStarted,
  type SessionEvent,
  type SessionId,
  ServingModelId,
} from "@magnitudedev/acn-protocol"
import {
  type ChatSessionRecord,
  ChatStore,
  ChatStoreCorrupt,
  ChatStoreLive,
  committedRecord,
  GlobalStorage,
  makeGlobalStorage,
  renamedRecord,
  renameOutcome,
} from "@magnitudedev/storage"
import { BunSqliteDriverLayer } from "@magnitudedev/storage/sqlite/bun"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import * as Native from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges } from "../changes"
import { InferenceActivity, InferenceActivityLive } from "../inference-activity"
import { ServingUsage, type ServingUsageRecord } from "../serving-usage"
import { ChatInference } from "./inference"
import { ChatService, ChatServiceLive } from "./service"

// Shared fixtures for the chat tests: events, a memory chat store, and the service over scripted inference.

export const at = (millis: number) => DateTime.unsafeMake(millis)
export const newId = () => Effect.runSync(generateMessageId)
export const newChat = () => Effect.runSync(generateSessionId)
export const model = ServingModelId.make("qwen-test")

export const settings: RequestedResponseSettings & ResponseSettings = {
  systemPrompt: "Be brief.",
  temperature: Option.some(0.7),
  maxOutputTokens: Option.some(256),
  topP: Option.none(),
  topK: Option.none(),
  minP: Option.none(),
  repetitionPenalty: Option.none(),
  presencePenalty: Option.none(),
  frequencyPenalty: Option.none(),
  seed: Option.some(7),
  stop: ["</end>"],
}

export const sent = (messageId: MessageId, text: string, time: number): MessageSent =>
  ({ _tag: "message_sent", time: at(time), messageId, text })

export const started = (responseId: MessageId, time: number, replaces = Option.none<MessageId>()): ResponseStarted => ({
  _tag: "response_started",
  time: at(time),
  responseId,
  replaces,
  model,
  reasoningEffort: Option.none(),
  settings,
})

export const output = (
  responseId: MessageId,
  part: "text" | "reasoning",
  time: number,
  deltas: readonly [string, ...string[]],
  offsets: readonly [number, ...number[]]
): SessionEvent => ({ _tag: "response_output", time: at(time), responseId, part, deltas, at: offsets })

export const finished = (responseId: MessageId, time: number, finishReason = "stop"): SessionEvent => ({
  _tag: "response_ended",
  time: at(time),
  responseId,
  outcome: { _tag: "finished", finishReason },
  usage: Option.none(),
  timings: Option.none(),
})

const parseError: ParseResult.ParseError = Effect.runSync(Effect.flip(Schema.decodeUnknown(ChatSummary)({})))

/** What the memory store fails with: stored data that does not decode. */
export const storeFailure = new ChatStoreCorrupt({ cause: parseError })

/**
 * A chat store in memory, keeping each session's record by the store's own rules. Its failures are
 * switched on by the test; a record or a log can be made undecodable.
 */
export const memoryChatStore = Effect.gen(function* () {
  const logs = new Map<SessionId, MemoryEventStore<SessionEvent>>()
  const records = new Map<SessionId, ChatSessionRecord>()
  /** Fails every operation, or only commits whose events match. */
  const failing = yield* Ref.make<(commit: Option.Option<ReadonlyArray<SessionEvent>>) => boolean>(() => false)
  /** Sessions whose record does not decode. */
  const unreadable = new Set<SessionId>()
  /** Sessions whose log does not decode. */
  const corrupt = new Set<SessionId>()
  /** What a delete waits for first. */
  const deleting = yield* Ref.make<Effect.Effect<void>>(Effect.void)
  const guard = <A, E>(
    commit: Option.Option<ReadonlyArray<SessionEvent>>,
    effect: Effect.Effect<A, E>
  ): Effect.Effect<A, E | ChatStoreCorrupt> =>
    Effect.flatMap(Ref.get(failing), (fails): Effect.Effect<A, E | ChatStoreCorrupt> =>
      fails(commit) ? Effect.fail(storeFailure) : effect)
  const logOf = (id: SessionId) => {
    const existing = logs.get(id)
    if (existing !== undefined) return existing
    const created = Effect.runSync(EventStore.memory<SessionEvent>())
    logs.set(id, created)
    return created
  }
  const append = (id: SessionId, events: ReadonlyArray<SessionEvent>) => Effect.zipRight(
    logOf(id).commit(events),
    Effect.sync(() => {
      Option.match(committedRecord(id, Option.fromNullable(records.get(id)), events), {
        onNone: () => undefined,
        onSome: (record) => records.set(id, record),
      })
    })
  )
  const rows = Effect.sync(() =>
    [...records.values()]
      .sort((a, b) => DateTime.toEpochMillis(b.time.updated) - DateTime.toEpochMillis(a.time.updated) || (a.id < b.id ? -1 : 1)))
  const store: ChatStore = {
    session: (id) => ({
      read: Stream.unwrap(guard(Option.none(), Effect.suspend(() =>
        corrupt.has(id) ? Effect.fail(storeFailure) : Effect.succeed(logOf(id).read)))),
      commit: (events) => guard(Option.some(events), append(id, events)),
    }),
    record: (id) => guard(Option.none(), Effect.suspend(() =>
      unreadable.has(id) ? Effect.fail(storeFailure) : Effect.succeed(Option.fromNullable(records.get(id))))),
    rename: (id, change) => guard(Option.none(), Effect.sync(() => {
      const record = records.get(id)
      const outcome = renameOutcome(Option.map(Option.fromNullable(record), ({ title }) => title), change)
      if (outcome === "renamed" && record !== undefined) records.set(id, renamedRecord(record, change))
      return outcome
    })),
    // A page holding a record that does not decode fails as corrupt, as the store's does.
    page: ({ offset, limit }) => guard(Option.none(), Effect.flatMap(rows, (all) => {
      const page = all.slice(offset, offset + limit)
      return page.some((record) => unreadable.has(record.id)) ? Effect.fail(storeFailure) : Effect.succeed(page)
    })),
    delete: (id) => Effect.zipRight(Effect.flatten(Ref.get(deleting)), guard(Option.none(), Effect.sync(() => {
      logs.delete(id)
      records.delete(id)
      unreadable.delete(id)
      corrupt.delete(id)
    }))),
  }
  return {
    store,
    failing,
    unreadable,
    corrupt,
    deleting,
    /** The session's events, or none once it is deleted. */
    events: (id: SessionId): Effect.Effect<Option.Option<ReadonlyArray<SessionEvent>>> => {
      const log = logs.get(id)
      return log === undefined ? Effect.succeed(Option.none()) : Effect.map(log.events, Option.some)
    },
    /** The session's record, as stored. */
    record: (id: SessionId): Option.Option<ChatSessionRecord> => Option.fromNullable(records.get(id)),
    /** Commits events straight to a session, as an earlier run left them. */
    seed: (id: SessionId, events: ReadonlyArray<SessionEvent>) => append(id, events),
  }
})
export type MemoryChatStore = Effect.Effect.Success<typeof memoryChatStore>

// Inference fixtures.

export const servedModel = {
  id: model,
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
export const imageModel = {
  ...servedModel,
  id: "image-test",
  architecture: { input_modalities: ["text"], output_modalities: ["image"] },
}
export const served = (data: ReadonlyArray<unknown>) =>
  Schema.decodeUnknownSync(Native.OpenAiModelsResponse)({ object: "list", data })

const chunk = (patch: Record<string, unknown>) =>
  Schema.decodeUnknownSync(Native.ChatCompletionChunk)({
    id: "chatcmpl-test",
    created: 0,
    model: "qwen-test-q4",
    object: "chat.completion.chunk",
    choices: [],
    ...patch,
  })
export const progress = (value: Record<string, unknown>) => chunk({ progress: value })
export const text = (content: string) => chunk({ choices: [{ index: 0, delta: { content } }] })
export const thought = (reasoning: string) => chunk({ choices: [{ index: 0, delta: { reasoning_content: reasoning } }] })
export const timings = {
  cache_n: 12,
  draft_n: 8,
  draft_n_accepted: 6,
  parser_ms: 1,
  predicted_ms: 50,
  predicted_n: 3,
  predicted_per_second: 60,
  predicted_per_token_ms: 16.67,
  prompt_ms: 10,
  // The prompt's tokens other than the 12 found in the cache: 40 in all.
  prompt_n: 28,
  prompt_per_second: 4000,
  prompt_per_token_ms: 0.25,
  sampler_ms: 2,
  time_to_first_token_ms: 15,
}
export const measured = () => chunk({ timings })
export const done = (finish = "stop") =>
  chunk({
    choices: [{ index: 0, delta: {}, finish_reason: finish }],
    timings,
  })

export type Generate = ChatInference["generate"]

/** The inference service, scripted, recording what it was asked. */
export const scriptedInference = (generate: Generate) => Effect.gen(function* () {
  const requests: Native.ChatCompletionRequest[] = []
  const models = yield* Ref.make(served([servedModel, imageModel]))
  const usage: ServingUsageRecord[] = []
  const layer = Layer.mergeAll(
    InferenceActivityLive,
    Layer.succeed(ServingUsage, {
      record: (record) => Effect.sync(() => { usage.push(record) }),
      read: () => Effect.die("Unused"),
    }),
    Layer.succeed(ChatInference, {
      models: Ref.get(models),
      generate: (request) => {
        requests.push(request)
        return generate(request)
      },
    })
  )
  return { layer, requests, models, usage }
})

/** The chat service over a store and scripted inference. */
const fixtureOver = (generate: Generate, store: Layer.Layer<ChatStore>) => Effect.gen(function* () {
  const inference = yield* scriptedInference(generate)
  const changes: Change[] = []
  const deps = Layer.mergeAll(
    store,
    Layer.succeed(AcnChanges, { publish: (change) => Effect.sync(() => { changes.push(change) }), stream: Stream.empty }),
    inference.layer
  )
  const layer = ChatServiceLive.pipe(Layer.provideMerge(deps))
  const run = <A, E>(effect: Effect.Effect<A, E, ChatService | ChatStore | InferenceActivity | Scope.Scope>) =>
    Effect.runPromise(Effect.scoped(effect.pipe(Effect.provide(layer))))
  return { run, changes, ...inference }
})

/** The chat service over a memory store and scripted inference. */
export const serviceFixture = (generate: Generate, existing?: MemoryChatStore) => Effect.gen(function* () {
  const memory = existing ?? (yield* memoryChatStore)
  return { ...(yield* fixtureOver(generate, Layer.succeed(ChatStore, memory.store))), memory }
})

/** The chat service over the chat store in `<root>/chats.sqlite` and scripted inference. */
export const sqliteServiceFixture = (generate: Generate, root: string) => fixtureOver(
  generate,
  ChatStoreLive.pipe(Layer.provide(Layer.mergeAll(
    BunSqliteDriverLayer,
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(GlobalStorage, makeGlobalStorage({ root }))
  )))
)

export const payload = (chatId: SessionId, value = "Hi") => ({
  chatId,
  messageId: newId(),
  responseId: newId(),
  text: value,
  model,
  reasoningEffort: Option.some("medium"),
  settings,
})
