import {
  Cause,
  Context,
  Deferred,
  Effect,
  Fiber,
  Layer,
  Option,
  Queue,
  Schema,
  Scope,
  Stream,
} from "effect"
import {
  applyChatGeneration,
  Chat,
  ChatActivity,
  ChatCommandId,
  ChatConversation,
  ChatConversationId,
  ChatFailed,
  ChatGeneration,
  ChatGenerationId,
  ChatMessageId,
  ChatSettings,
  ChatUpdate,
  emptyChatMetrics,
  ServingModelId,
} from "@magnitudedev/acn-protocol"
import { ChatStorage, type StoredChatGeneration } from "@magnitudedev/storage"
import * as Native from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges } from "../changes"
import { ServingUsage, ServingUsageRecord } from "../serving-usage"
import { InferenceActivity } from "../inference-activity"
import { ChatInference } from "./inference"
import magnitudeInfo from "./magnitude-info.md" with { type: "text" }

export interface ChatService {
  readonly models: ChatInference["models"]
  readonly list: ChatStorage["list"]
  readonly get: ChatStorage["get"]
  readonly create: ChatStorage["create"]
  readonly update: ChatStorage["update"]
  readonly remove: ChatStorage["remove"]
  readonly start: (input: {
    readonly conversationId: ChatConversationId
    readonly commandId: ChatCommandId
    readonly content: Option.Option<string>
    readonly retryMessageId: Option.Option<ChatMessageId>
    readonly settings: ChatSettings
  }) => Effect.Effect<ChatGenerationId, ChatFailed>
  readonly stop: (
    conversationId: ChatConversationId,
    generationId: ChatGenerationId
  ) => Effect.Effect<void, ChatFailed>
  readonly watch: (
    id: ChatConversationId
  ) => Stream.Stream<ChatUpdate, ChatFailed>
}
export const ChatService = Context.GenericTag<ChatService>(
  "@magnitudedev/acn/ChatService"
)
interface Running {
  value: StoredChatGeneration
  fiber: Option.Option<Fiber.RuntimeFiber<void, never>>
  dirty: boolean
  lastSave: number
  reasoningStarted: Option.Option<number>
  completion: Option.Option<"completed" | "length">
  stopRequested: boolean
}
interface Subscriber {
  readonly queue: Queue.Queue<void>
  pending: Option.Option<ChatUpdate>
}
/** Keep a pending structural snapshot while folding later output into it. */
const offer = (subscriber: Subscriber, update: ChatUpdate) => Effect.sync(() => {
  const pending = Option.getOrUndefined(subscriber.pending)
  subscriber.pending = Option.some(update._tag === "Generation" && pending?._tag === "Snapshot"
    ? { _tag: "Snapshot", conversation: applyChatGeneration(pending.conversation, update.generation) }
    : update._tag === "Generation" && pending?._tag === "Deleted" ? pending : update)
}).pipe(Effect.zipRight(Queue.offer(subscriber.queue, undefined)), Effect.asVoid)

const invalid = (message: string) =>
  new ChatFailed({ code: "InvalidRequest", message })
const someFinite = (n: number) =>
  Number.isFinite(n) && n >= 0 ? Option.some(n) : Option.none<number>()

export const ChatServiceLive = Layer.scoped(
  ChatService,
  Effect.gen(function* () {
    const storage = yield* ChatStorage
    const inference = yield* ChatInference
    const changes = yield* AcnChanges
    const usage = yield* ServingUsage
    const activity = yield* InferenceActivity
    const scope = yield* Scope.Scope
    const lock = yield* Effect.makeSemaphore(1)
    const deleting = new Set<ChatConversationId>()
    const running = new Map<ChatConversationId, Running>()
    const viewers = new Map<ChatConversationId, Set<Subscriber>>()
    const historyChanged = changes.publish({
      operation: Chat.listConversations._tag,
    })
    const get = (id: ChatConversationId) =>
      storage.get(id).pipe(
        Effect.map((conversation) => {
          const live = running.get(id)?.value
          return !live
            ? conversation
            : {
                ...conversation,
                updatedAt: live.updatedAt,
                messages: conversation.messages.map((m) =>
                  m.id === live.messageId
                    ? {
                        ...m,
                        content: live.content,
                        generation: Option.some(live),
                      }
                    : m
                ),
              }
        })
      )
    const publish = (id: ChatConversationId) =>
      get(id).pipe(
        Effect.flatMap((conversation) =>
          Effect.forEach(
            viewers.get(id) ?? [],
            (q) => offer(q, { _tag: "Snapshot", conversation }),
            { discard: true }
          )
        ),
        Effect.asVoid
      )
    const publishGeneration = (id: ChatConversationId, generation: ChatGeneration) => Effect.forEach(
      viewers.get(id) ?? [], subscriber => offer(subscriber, { _tag: "Generation", generation }), { discard: true },
    )
    const snapshotMutation = <A>(
      id: ChatConversationId,
      operation: Effect.Effect<A, ChatFailed>
    ) =>
      lock.withPermits(1)(
        operation.pipe(
          Effect.tap(() => publish(id)),
          Effect.tap(() => historyChanged)
        )
      )
    const setPhase = (
      entry: Running,
      phase: (typeof ChatActivity.Type)["phase"],
      details: Partial<ChatActivity> = {}
    ) => {
      const previous = Option.getOrUndefined(entry.value.activity)
      entry.value = {
        ...entry.value,
        activity: Option.some({
          phase,
          since: previous?.phase === phase ? previous.since : Date.now(),
          fraction: Option.none(),
          completed: Option.none(),
          total: Option.none(),
          cached: Option.none(),
          ...details,
        }),
      }
    }
    const consume = (entry: Running, chunk: Native.ChatCompletionChunk) =>
      Effect.gen(function* () {
        const now = Date.now()
        if (Option.isSome(chunk.progress)) {
          const p = chunk.progress.value
          if (p.phase === "model_loading")
            setPhase(
              entry,
              p.stage === "optimizing"
                ? "optimizing"
                : p.stage === "finalizing"
                ? "finalizing"
                : p.stage === "queued"
                ? "queued"
                : p.stage === "preparing"
                ? "preparing"
                : "loading",
              { fraction: someFinite(Math.min(1, Math.max(0, p.fraction))) }
            )
          else if (p.phase === "prefill")
            setPhase(entry, "prefill", {
              fraction:
                p.total_tokens > 0
                  ? Option.some(
                      Math.min(1, p.completed_tokens / p.total_tokens)
                    )
                  : Option.none(),
              completed: Option.some(p.completed_tokens),
              total: Option.some(p.total_tokens),
              cached: Option.some(p.cached_tokens),
            })
          else if (p.phase !== "generating") setPhase(entry, p.phase)
        }
        if (Option.isSome(chunk.timings)) {
          const t = chunk.timings.value
          entry.value = {
            ...entry.value,
            metrics: {
              ...entry.value.metrics,
              inputTokens: Option.some(t.prompt_n),
              cachedTokens: Option.some(t.cache_n),
              outputTokens: Option.some(t.predicted_n),
              decodeMs: someFinite(t.predicted_ms),
              tokensPerSecond: someFinite(t.predicted_per_second),
              promptMs: someFinite(t.prompt_ms),
              promptTokensPerSecond: someFinite(t.prompt_per_second),
              engineTtftMs: someFinite(t.time_to_first_token_ms),
            },
          }
        }
        if (Option.isSome(chunk.usage)) {
          const u = chunk.usage.value
          entry.value = {
            ...entry.value,
            metrics: {
              ...entry.value.metrics,
              inputTokens: Option.some(u.prompt_tokens),
              outputTokens: Option.some(u.completion_tokens),
              cachedTokens: Option.some(u.prompt_tokens_details.cached_tokens),
            },
          }
        }
        const choice = chunk.choices.find((c) => c.index === 0)
        if (choice) {
          const content = Option.getOrNull(choice.delta.content) ?? ""
          const reasoning = Option.getOrElse(
            choice.delta.reasoning_content,
            () => ""
          )
          if (content || reasoning) {
            let metrics = entry.value.metrics
            if (Option.isNone(metrics.ttftMs))
              metrics = {
                ...metrics,
                ttftMs: Option.some(now - entry.value.createdAt),
              }
            if (reasoning && Option.isNone(entry.reasoningStarted))
              entry.reasoningStarted = Option.some(now)
            if (content && Option.isNone(metrics.firstAnswerMs))
              metrics = {
                ...metrics,
                firstAnswerMs: Option.some(now - entry.value.createdAt),
                reasoningMs: Option.map(
                  entry.reasoningStarted,
                  (start) => now - start
                ),
              }
            entry.value = {
              ...entry.value,
              content: entry.value.content + content,
              reasoning: entry.value.reasoning + reasoning,
              metrics,
            }
            setPhase(entry, content ? "responding" : "thinking")
          }
          if (Option.isSome(choice.finish_reason)) {
            const reason = choice.finish_reason.value
            if (reason !== "stop" && reason !== "length")
              return yield* new ChatFailed({
                code: "Inference",
                message: `The model ended with ${reason}. This chat has no tools.`,
              })
            entry.completion = Option.some(
              reason === "length" ? "length" : "completed"
            )
          }
        }
        entry.value = { ...entry.value, updatedAt: now }
        entry.dirty = true
        if (now - entry.lastSave >= 1000) {
          yield* storage.save(entry.value)
          entry.lastSave = now
        }
      })
    const finish = (
      id: ChatConversationId,
      entry: Running,
      cause: Option.Option<Cause.Cause<ChatFailed>>
    ) =>
      lock.withPermits(1)(
        Effect.gen(function* () {
          const now = Date.now()
          const status = Option.isSome(cause)
            ? Cause.isInterruptedOnly(cause.value)
              ? entry.stopRequested
                ? "stopped"
                : "interrupted"
              : "failed"
            : Option.getOrElse(entry.completion, () => "interrupted" as const)
          const error =
            Option.isSome(cause) && !Cause.isInterruptedOnly(cause.value)
              ? Option.some(
                  Option.getOrElse(
                    Cause.failureOption(cause.value),
                    () =>
                      new ChatFailed({
                        code: "Inference",
                        message:
                          "The response was interrupted by an unexpected service error.",
                      })
                  ).message
                )
              : status === "interrupted"
              ? Option.some(
                  "The response ended before completion. Retry to continue."
                )
              : Option.none<string>()
          entry.value = {
            ...entry.value,
            status,
            error,
            updatedAt: now,
            activity: Option.none(),
            metrics: {
              ...entry.value.metrics,
              totalMs: Option.some(now - entry.value.createdAt),
              reasoningMs: Option.orElse(entry.value.metrics.reasoningMs, () =>
                Option.map(entry.reasoningStarted, (start) => now - start)
              ),
            },
          }
          const saved = yield* Effect.either(storage.save(entry.value))
          if (saved._tag === "Left") {
            entry.value = {
              ...entry.value,
              error: Option.some(
                "The response ended, but chat history could not be saved. Check access to the Magnitude data folder before retrying."
              ),
            }
            yield* Effect.logError(saved.left.message)
          }
          yield* publishGeneration(id, entry.value)
          // Retain an unsaved terminal response for this process and retry its write on the next send.
          if (saved._tag === "Right") running.delete(id)
          yield* historyChanged
          const m = entry.value.metrics
          yield* usage.record({
            id: ServingUsageRecord.fields.id.make(`chat:${entry.value.id}`),
            model: ServingModelId.make(entry.value.modelId),
            completedAt: now,
            input: Option.getOrNull(m.inputTokens),
            cached: Option.getOrNull(m.cachedTokens),
            output: Option.getOrNull(m.outputTokens),
            generationMs: Option.getOrNull(
              Option.filter(m.decodeMs, (n) => n > 0)
            ),
            firstTokenMs: Option.getOrNull(m.ttftMs),
            complete: status === "completed" || status === "length",
          })
        })
      )
    const start: ChatService["start"] = (input) =>
      lock.withPermits(1)(
        Effect.uninterruptible(
          Effect.gen(function* () {
            const previous = yield* storage.findCommand(
              input.commandId,
              input.conversationId
            )
            if (Option.isSome(previous)) return previous.value.id
            if (
              running.get(input.conversationId)?.value.status === "running" ||
              deleting.has(input.conversationId)
            )
              return yield* new ChatFailed({
                code: "Busy",
                message: "A response is already running in this conversation.",
              })
            if (
              Option.isSome(input.content) ===
              Option.isSome(input.retryMessageId)
            )
              return yield* invalid(
                "Send a message or retry the last response."
              )
            if (Option.isSome(input.content) && !input.content.value.trim())
              return yield* invalid("Enter a message first.")
            const unsaved = running.get(input.conversationId)
            if (unsaved) {
              yield* storage.save(unsaved.value)
              running.delete(input.conversationId)
            }
            const conversation = yield* storage.get(input.conversationId)
            const models = yield* inference.models
            const modelId = yield* Option.match(input.settings.modelId, {
              onNone: () => Effect.fail(invalid("Choose a model first.")),
              onSome: Effect.succeed,
            })
            const model = models.data.find((m) => m.id === modelId)
            if (
              !model ||
              !model.architecture.input_modalities.includes("text") ||
              !model.architecture.output_modalities.includes("text")
            )
              return yield* invalid(
                "This model is unavailable for text chat. Choose an installed text model."
              )
            const reasoning = Option.getOrNull(model.reasoning)
            if (
              Option.isSome(input.settings.reasoningEffort) &&
              (!reasoning?.supported_efforts.includes(
                input.settings.reasoningEffort.value
              ) ||
                (reasoning.mandatory &&
                  ["none", "off"].includes(
                    input.settings.reasoningEffort.value
                  )))
            )
              return yield* invalid(
                "This model does not support that reasoning level."
              )
            const setup = (yield* inference.setup).slice(0, 32768)
            const settings = input.settings
            const optional = <A>(value: Option.Option<A>) =>
              Option.getOrUndefined(value)
            const messages = Option.isSome(input.retryMessageId)
              ? conversation.messages.slice(0, -1)
              : conversation.messages
            const request = yield* Schema.decodeUnknown(
              Native.ChatCompletionRequest
            )({
              model: modelId,
              stream: true,
              stream_options: { include_usage: true },
              timings_per_token: true,
              messages: [
                {
                  role: "system",
                  content: `You are a helpful assistant. Answer the user directly. You have no tools or independent access to their machine.\n\n${settings.systemPrompt}\n\n${magnitudeInfo}\n\n## Current setup\n${setup}\n\nSelected model: ${model.name} (${modelId}). Context: ${model.context_length} tokens. No tools are available.`,
                },
                ...messages
                  .filter((m) => m.role === "user" || m.content.length > 0)
                  .map((m) => ({ role: m.role, content: m.content })),
                ...(Option.isSome(input.content)
                  ? [{ role: "user", content: input.content.value }]
                  : []),
              ],
              ...Object.fromEntries(
                Object.entries({
                  temperature: optional(settings.temperature),
                  max_tokens: optional(settings.maxOutputTokens),
                  reasoning_effort: optional(settings.reasoningEffort),
                  top_p: optional(settings.topP),
                  top_k: optional(settings.topK),
                  min_p: optional(settings.minP),
                  repetition_penalty: optional(settings.repetitionPenalty),
                  presence_penalty: optional(settings.presencePenalty),
                  frequency_penalty: optional(settings.frequencyPenalty),
                  seed: optional(settings.seed),
                  stop: settings.stop.length ? settings.stop : undefined,
                }).filter(([, v]) => v !== undefined)
              ),
            }).pipe(
              Effect.mapError(() =>
                invalid("These settings could not be used for this model.")
              )
            )
            const now = Date.now()
            const generation: StoredChatGeneration = {
              id: ChatGenerationId.make(crypto.randomUUID()),
              commandId: input.commandId,
              messageId: Option.getOrElse(input.retryMessageId, () =>
                ChatMessageId.make(crypto.randomUUID())
              ),
              modelId,
              settings,
              status: "running",
              createdAt: now,
              updatedAt: now,
              content: "",
              reasoning: "",
              activity: Option.some({
                phase: "queued",
                since: now,
                fraction: Option.none(),
                completed: Option.none(),
                total: Option.none(),
                cached: Option.none(),
              }),
              metrics: emptyChatMetrics,
              error: Option.none(),
              requestJson: yield* Schema.encode(
                Schema.parseJson(
                  Schema.Struct({
                    infoVersion: Schema.Literal(1),
                    request: Native.ChatCompletionRequest,
                  })
                )
              )({ infoVersion: 1, request }).pipe(Effect.orDie),
            }
            yield* storage.admit(
              input.conversationId,
              generation,
              Option.map(input.content, (content) => ({
                id: ChatMessageId.make(crypto.randomUUID()),
                content,
              })),
              input.retryMessageId
            )
            const entry: Running = {
              value: generation,
              fiber: Option.none(),
              dirty: false,
              lastSave: now,
              reasoningStarted: Option.none(),
              completion: Option.none(),
              stopRequested: false,
            }
            running.set(input.conversationId, entry)
            yield* publish(input.conversationId).pipe(Effect.catchAll(error => Effect.logError(error.message)))
            yield* historyChanged
            const releaseActivity = yield* activity.begin
            const ready = yield* Deferred.make<void>()
            const work = Effect.uninterruptibleMask((restore) =>
              Deferred.succeed(ready, undefined).pipe(
                Effect.zipRight(
                  restore(Effect.scoped(Effect.gen(function* () {
                    yield* Effect.forkScoped(Effect.forever(Effect.sleep("80 millis").pipe(
                      Effect.zipRight(lock.withPermits(1)(Effect.suspend(() => {
                        if (!entry.dirty) return Effect.void
                        entry.dirty = false
                        return publishGeneration(input.conversationId, entry.value)
                      }))),
                    )))
                    yield* inference.generate(request).pipe(Stream.runForEach(chunk => consume(entry, chunk)))
                  }))                  ).pipe(
                    Effect.exit,
                    Effect.flatMap((exit) =>
                      finish(
                        input.conversationId,
                        entry,
                        exit._tag === "Failure"
                          ? Option.some(exit.cause)
                          : Option.none()
                      )
                    )
                  )
                )
              )
            )
            entry.fiber = Option.some(
              yield* Effect.forkIn(
                work.pipe(
                  Effect.interruptible,
                  Effect.ensuring(Effect.sync(releaseActivity))
                ),
                scope
              )
            )
            yield* Deferred.await(ready)
            return generation.id
          })
        )
      )
    return ChatService.of({
      models: inference.models,
      list: (offset, limit) =>
        storage.list(offset, limit).pipe(
          Effect.map((history) => ({
            ...history,
            conversations: history.conversations.map((c) => {
              const entry = running.get(c.id)
              return entry
                ? { ...c, active: entry.value.status === "running" }
                : c
            }),
          }))
        ),
      get,
      create: (id, settings) =>
        snapshotMutation(id, storage.create(id, settings)),
      update: (id, title, settings) =>
        snapshotMutation(id, storage.update(id, title, settings)),
      remove: (id) =>
        Effect.uninterruptible(
          Effect.gen(function* () {
            const active = yield* lock.withPermits(1)(
              Effect.sync(() => {
                deleting.add(id)
                return running.get(id)
              })
            )
            if (active && Option.isSome(active.fiber))
              yield* Fiber.interrupt(active.fiber.value)
            yield* lock.withPermits(1)(
              storage.remove(id).pipe(
                Effect.tap(() =>
                  Effect.sync(() => {
                    running.delete(id)
                  })
                ),
                Effect.tap(() =>
                  Effect.forEach(
                    viewers.get(id) ?? [],
                    (q) =>
                      offer(q, { _tag: "Deleted", conversationId: id }),
                    { discard: true }
                  )
                ),
                Effect.tap(() => historyChanged)
              )
            )
          }).pipe(
            Effect.ensuring(
              Effect.sync(() => {
                deleting.delete(id)
              })
            )
          )
        ),
      start,
      stop: (id, generationId) =>
        Effect.gen(function* () {
          const active = running.get(id)
          if (
            active?.value.id === generationId &&
            Option.isSome(active.fiber)
          ) {
            active.stopRequested = true
            yield* Fiber.interrupt(active.fiber.value)
          }
        }),
      watch: id => Stream.unwrapScoped(Effect.gen(function* () {
        const queue = yield* Effect.acquireRelease(Queue.sliding<void>(1), Queue.shutdown)
        const subscriber: Subscriber = { queue, pending: Option.none() }
        const initial = yield* lock.withPermits(1)(Effect.gen(function* () {
          const conversation = yield* get(id)
          const set = viewers.get(id) ?? new Set<Subscriber>()
          set.add(subscriber); viewers.set(id, set)
          return { _tag: "Snapshot" as const, conversation }
        }))
        yield* Effect.addFinalizer(() => Effect.sync(() => {
          viewers.get(id)?.delete(subscriber)
          if (viewers.get(id)?.size === 0) viewers.delete(id)
        }))
        return Stream.concat(Stream.succeed(initial), Stream.fromQueue(queue).pipe(
          Stream.mapEffect(() => Effect.sync(() => {
            const pending = subscriber.pending
            subscriber.pending = Option.none()
            return pending
          })),
          Stream.filterMap(update => update),
        ))
      })),
    })
  })
)
