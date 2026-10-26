import { Cause, Context, Data, DateTime, Effect, type Either, Equal, Layer, Option, Ref, Stream, SubscriptionRef } from "effect"
import type { Closure, EngineClosed, Emit, EventSettlementError, ProjectionReader } from "@magnitudedev/event-engine"
import {
  Chat,
  type ChatBusy,
  type ChatMessage,
  type ChatMessagePayload,
  ChatNotFound,
  type ChatPage,
  type ChatRenamePayload,
  type ChatRetryPayload,
  type ChatStopPayload,
  ChatStorageFailed,
  type ChatSummary,
  type ChatUpdate,
  InvalidChatRequest,
  latestResponse,
  type LiveResponse,
  type ServableModelsUnavailable,
  type SessionEvent,
  type SessionId,
} from "@magnitudedev/acn-protocol"
import { ChatStore, type ChatSessionRecord, type ChatStoreError, committedRecord, isChatStoreError, renamedRecord } from "@magnitudedev/storage"
import { AcnChanges } from "../changes"
import * as Commands from "./commands"
import { ChatEngine, settled, storageFailed } from "./engine"
import { lastObserved, Live, runningResponse, Seen, Transcript } from "./folds"
import { Generation } from "./generation"
import { ChatInference } from "./inference"
import { makeInstances } from "./instances"

export interface ChatService {
  readonly list: (offset: number, limit: number) => Effect.Effect<ChatPage, ChatStorageFailed>
  readonly start: (
    input: ChatMessagePayload
  ) => Effect.Effect<void, InvalidChatRequest | ServableModelsUnavailable | ChatStorageFailed>
  readonly send: (
    input: ChatMessagePayload
  ) => Effect.Effect<void, ChatNotFound | ChatBusy | InvalidChatRequest | ServableModelsUnavailable | ChatStorageFailed>
  readonly retry: (
    input: ChatRetryPayload
  ) => Effect.Effect<void, ChatNotFound | ChatBusy | InvalidChatRequest | ServableModelsUnavailable | ChatStorageFailed>
  readonly rename: (input: ChatRenamePayload) => Effect.Effect<void, ChatNotFound | InvalidChatRequest | ChatStorageFailed>
  readonly stop: (input: ChatStopPayload) => Effect.Effect<void, ChatNotFound | InvalidChatRequest | ChatStorageFailed>
  readonly remove: (chatId: SessionId) => Effect.Effect<void, ChatStorageFailed>
  readonly watch: (chatId: SessionId) => Stream.Stream<ChatUpdate, ChatNotFound | ChatStorageFailed>
}
export const ChatService = Context.GenericTag<ChatService>("@magnitudedev/acn/ChatService")

export const ChatEngineDefinition = ChatEngine.make({
  name: "chat",
  projections: [Seen, Transcript, Live],
  workers: [Generation],
})

/** A chat as it is listed and watched: its session's record, and whether its transcript has a response running. */
const summaryOf = (record: ChatSessionRecord, transcript: ReadonlyArray<ChatMessage>): ChatSummary =>
  Data.struct({ ...record, time: Data.struct(record.time), running: Option.isSome(runningResponse(transcript)) })

/**
 * A response found running when its session opens belonged to a runtime that is gone: a session has
 * one instance at a time, which closes once idle, or when it fails, leaving the response it ran to
 * the session's next open. It ended when it was last observed. Returns whether it ended one.
 */
const recover = ({ read, emit }: { readonly read: ProjectionReader; readonly emit: Emit<SessionEvent, ChatStoreError> }) =>
  Effect.flatMap(read.get(Transcript), (transcript) =>
    Option.match(Option.filter(latestResponse(transcript), (response) => Option.isNone(response.end)), {
      onNone: () => Effect.succeed(false),
      onSome: (response) => Effect.as(emit({
        _tag: "response_ended",
        time: lastObserved(response),
        responseId: response.id,
        outcome: { _tag: "stopped", by: "shutdown" },
        usage: Option.none(),
        timings: Option.none(),
      }), true),
    }))

/** Whether two transcripts differ only in output: the same messages, and the same responses ended. */
const listedAlike = (left: ReadonlyArray<ChatMessage>, right: ReadonlyArray<ChatMessage>) =>
  left.length === right.length && left.every((message, index) => {
    const other = right[index]!
    return message.id === other.id && (message._tag === "user" || (other._tag === "assistant" && Option.isSome(message.end) === Option.isSome(other.end)))
  })

/**
 * A chat's commands, as transactions, and what is followed of it: its settled transcript and live
 * state, and with them its record, read in the same step, so a commit is in both or in neither.
 */
const ChatFacade = ChatEngine.Facade.define({
  name: "Chat",
  mutations: ({ transact }) => {
    const command = <J>(body: Commands.Command<J>) => transact(({ read }) => body(read))
    return {
      start: (input: ChatMessagePayload) => command(Commands.start(input)),
      send: (input: ChatMessagePayload) => command(Commands.send(input)),
      retry: (input: ChatRetryPayload) => command(Commands.retry(input)),
      stop: (input: ChatStopPayload) => command(Commands.stop(input)),
    }
  },
  queries: ({ read }) => ({
    states: () => settled(Commands.states(read)),
    transcript: () => settled(read.get(Transcript)),
    view: () => settled(Effect.map(Effect.all({ messages: read.get(Transcript), live: read.get(Live) }), Data.struct)),
    watched: (record: Effect.Effect<Option.Option<ChatSessionRecord>>) => settled(Effect.map(
      Effect.all({ record, messages: read.get(Transcript), live: read.get(Live) }),
      ({ record, messages, live }): View => Data.struct({ chat: Option.map(record, (found) => summaryOf(found, messages)), messages, live })
    )),
  }),
})

interface View {
  readonly chat: Option.Option<ChatSummary>
  readonly messages: ReadonlyArray<ChatMessage>
  readonly live: Option.Option<LiveResponse>
}

/**
 * The one update that takes a watcher from one view to the next: every part that changed, by
 * identity. A view is read in one step, so its parts are those of one settled state.
 */
const changeBetween = (previous: View, current: View): ChatUpdate => ({
  _tag: "changed",
  chat: Equal.equals(previous.chat, current.chat) ? Option.none() : current.chat,
  messages: current.messages.flatMap((message, index) => previous.messages[index] === message ? [] : [{ index, message }]),
  live: Equal.equals(previous.live, current.live) ? Option.none() : Option.some(current.live),
})

/** A failed instance's failure, as clients are told it: the store's, or a defect. */
const failedWith = (closure: Extract<Closure<ChatStoreError>, { _tag: "Failed" }>): Effect.Effect<never, ChatStorageFailed> =>
  Option.match(Cause.failureOption(closure.cause), {
    onSome: (error) => isChatStoreError(error) ? Effect.fail(storageFailed(error)) : Effect.die(error),
    onNone: () => Effect.failCause(closure.cause as Cause.Cause<never>),
  })

/**
 * An open session: its engine, its commands and queries, and its record. The record changes only by
 * the session's own commits, all made by this instance, and by renames, so the instance loads it
 * once and applies each of those to it as the store does.
 */
interface ChatInstance {
  readonly record: SubscriptionRef.SubscriptionRef<Option.Option<ChatSessionRecord>>
  readonly closed: Effect.Effect<Closure<ChatStoreError>>
  /** Once the instance has closed as removed: the removal's delete, as clients are told it. */
  readonly removed: Effect.Effect<void, ChatStorageFailed>
  readonly facade: ReturnType<typeof ChatFacade.make>
  readonly awaitIdle: Effect.Effect<void>
  readonly close: (reason: "closed" | "removed") => Effect.Effect<void>
  readonly failed: Effect.Effect<boolean>
}

export const ChatServiceLive = Layer.scoped(
  ChatService,
  Effect.gen(function* () {
    const store = yield* ChatStore
    const changes = yield* AcnChanges
    const inference = yield* ChatInference
    const listChanged = changes.publish({ operation: Chat.list._tag })

    const instances = yield* makeInstances({
      open: (chatId: SessionId, removed: Effect.Effect<void, ChatStoreError>) => Effect.gen(function* () {
        const record = yield* SubscriptionRef.make(yield* store.record(chatId))
        const log = store.session(chatId)
        // Recovery runs before anything can close the instance it opens, and the chat list is told
        // of the response it ends. Each commit, once stored, applies to the record held here; it runs
        // under the engine's lock, so a view sees a commit in the record and the transcript together.
        const runtime = yield* settled(ChatEngineDefinition.open({
          store: {
            read: log.read,
            commit: (events) => Effect.zipRight(log.commit(events), SubscriptionRef.update(record, (current) => committedRecord(chatId, current, events))),
          },
          recover: (context) => Effect.tap(recover(context), (ended) => ended ? listChanged : Effect.void),
        }))
        const facade = ChatFacade.make({ runtime })
        // The chat list follows every change to the transcript other than output, which changes
        // the record's caches or whether a response runs, from the transcript the session opened
        // with, read here, before any command can change it.
        const opened = yield* facade.queries.transcript.get()
        yield* Effect.forkScoped(facade.queries.transcript.changes().pipe(
          Stream.dropWhile((transcript) => listedAlike(transcript, opened)),
          Stream.changesWith(listedAlike),
          Stream.runForEach(() => listChanged)
        ))
        return {
          record,
          closed: runtime.closed,
          removed: Effect.mapError(removed, storageFailed),
          facade,
          awaitIdle: runtime.awaitIdle,
          close: runtime.close,
          failed: Effect.map(runtime.closed, (closure) => closure._tag === "Failed"),
        } satisfies ChatInstance
      }),
      remove: (chatId: SessionId) => store.delete(chatId),
      // A failed instance may have left a response running, which its session's next open ends; it
      // no longer runs, so the chat list, told of it, lists it as not running.
      failed: () => listChanged,
      idleTimeToLive: "30 seconds",
    })

    /** Runs with the chat's instance. */
    const withChat = <A, E>(chatId: SessionId, use: (instance: ChatInstance) => Effect.Effect<A, E>) =>
      Effect.scoped(Effect.flatMap(Effect.mapError(instances.get(chatId), storageFailed), use))

    /**
     * Commits what a command decided, or tells why it could not. A command racing the chat's removal
     * finds it gone once the removal's delete has succeeded, and fails for the store's reason if the
     * delete failed; one racing a store failure fails for the store's reason.
     */
    const decide = <J>(
      instance: ChatInstance,
      command: Effect.Effect<Either.Either<void, J>, EventSettlementError | ChatStoreError | EngineClosed>,
      gone: () => J
    ) =>
      settled(command).pipe(
        Effect.catchAll((error): Effect.Effect<never, J | ChatStorageFailed> => error._tag === "EngineClosed"
          ? Effect.flatMap(instance.closed, (closure): Effect.Effect<never, J | ChatStorageFailed> => {
              switch (closure._tag) {
                case "Failed":
                  return failedWith(closure)
                case "Removed":
                  return Effect.zipRight(instance.removed, Effect.fail(gone()))
                case "Closed":
                  return Effect.interrupt
              }
            })
          : Effect.fail(storageFailed(error))),
        Effect.flatten
      )

    const notFound = (chatId: SessionId) => () => new ChatNotFound({ chatId })

    /** The model must be served for text, with the reasoning effort it supports. */
    const validate = ({ model, reasoningEffort }: ChatMessagePayload | ChatRetryPayload) => Effect.gen(function* () {
      const served = yield* inference.models
      const found = served.data.find((candidate) => candidate.id === model)
      if (
        found === undefined ||
        !found.architecture.input_modalities.includes("text") ||
        !found.architecture.output_modalities.includes("text")
      ) {
        return yield* new InvalidChatRequest({ reason: "model_unavailable" })
      }
      if (Option.isSome(reasoningEffort)) {
        const reasoning = Option.getOrNull(found.reasoning)
        if (
          reasoning === null ||
          !reasoning.supported_efforts.includes(reasoningEffort.value) ||
          (reasoning.mandatory && ["none", "off"].includes(reasoningEffort.value))
        ) {
          return yield* new InvalidChatRequest({ reason: "reasoning_unsupported" })
        }
      }
    })

    /**
     * Validates a command's model, unless its chat does not exist, which is told first, or the
     * command already took effect, then decides it.
     */
    const unlessReplayed = <J>(
      input: ChatMessagePayload | ChatRetryPayload,
      replayed: (states: Commands.States) => boolean,
      command: (mutations: ChatInstance["facade"]["mutations"]) => Effect.Effect<
        Either.Either<void, J>,
        EventSettlementError | ChatStoreError | EngineClosed
      >
    ) => withChat(input.chatId, (instance) => Effect.gen(function* () {
      const states = yield* instance.facade.queries.states.get()
      if (!states.exists) return yield* new ChatNotFound({ chatId: input.chatId })
      if (!replayed(states)) yield* validate(input)
      yield* decide<J | ChatNotFound>(instance, command(instance.facade.mutations), notFound(input.chatId))
    }))

    /**
     * A record as listed: a response runs only in its session's open instance, so a session with
     * none has no response running. The list is told whenever that changes: of every change to an
     * open chat, of recovery's end, of an instance that failed once it is forgotten, of a rename, and
     * of a removal once it ends, whether or not its delete succeeded.
     */
    const listed = (record: ChatSessionRecord): Effect.Effect<ChatSummary> =>
      Effect.flatMap(instances.current(record.id), Option.match({
        onNone: () => Effect.succeed(summaryOf(record, [])),
        onSome: (instance) => Effect.map(instance.facade.queries.transcript.get(), (transcript) => summaryOf(record, transcript)),
      }))

    return ChatService.of({
      list: (offset, limit) =>
        // One more row than asked for says whether there are more.
        Effect.mapError(store.page({ offset, limit: limit + 1 }), storageFailed).pipe(
          Effect.flatMap((records) => Effect.map(
            Effect.forEach(records.slice(0, limit), listed),
            (chats): ChatPage => ({ chats, hasMore: records.length > limit })
          ))
        ),
      // Only a start creates a chat. Its IDs are the client's, so a chat that exists is either this
      // start's, replayed, or another's. A removed chat is gone, so a replay after it recreates it.
      start: (input) => withChat(input.chatId, (instance) => Effect.gen(function* () {
        if (!Commands.startReplayed(input)(yield* instance.facade.queries.states.get())) yield* validate(input)
        yield* decide(instance, instance.facade.mutations.start(input), () => new InvalidChatRequest({ reason: "ids_in_use" }))
      })),
      send: (input) => unlessReplayed(input, Commands.sendReplayed(input), (mutations) => mutations.send(input)),
      retry: (input) => unlessReplayed(input, Commands.retryReplayed(input), (mutations) => mutations.retry(input)),
      /**
       * Renames from the title the client saw, in the record. A chat that already has the title is
       * renamed already, as by a replay; one whose title is neither has been renamed meanwhile,
       * which the client is told. A rename applies to the record of the chat's instance if one is
       * open or opening: an instance that opens after the store's rename loads the renamed record,
       * and applying it again changes nothing.
       */
      rename: ({ chatId, from, title }) => Effect.flatMap(DateTime.now, (time) => Effect.mapError(store.rename(chatId, { from, title, time }), storageFailed).pipe(
        Effect.flatMap((outcome): Effect.Effect<void, ChatNotFound | InvalidChatRequest> => {
          switch (outcome) {
            case "renamed":
              return instances.current(chatId).pipe(
                Effect.flatMap(Option.match({
                  onNone: () => Effect.void,
                  onSome: (instance) => SubscriptionRef.update(instance.record, Option.map((record) => renamedRecord(record, { title, time }))),
                })),
                Effect.zipRight(listChanged)
              )
            case "unchanged":
              return Effect.void
            case "title_changed":
              return Effect.fail(new InvalidChatRequest({ reason: "title_changed" }))
            case "missing":
              return Effect.fail(new ChatNotFound({ chatId }))
          }
        })
      )),
      stop: (input) => withChat(input.chatId, (instance) =>
        decide(instance, instance.facade.mutations.stop(input), notFound(input.chatId))),
      remove: (chatId) => instances.remove(chatId).pipe(Effect.mapError(storageFailed), Effect.ensuring(listChanged)),
      watch: (chatId) => Stream.unwrapScoped(Effect.gen(function* () {
        const { closed, removed, facade, record } = yield* Effect.mapError(instances.get(chatId), storageFailed)
        // Each change to the transcript or live state, and each change to the record the instance
        // holds (a rename: a watch holds its chat's instance, so every rename of it lands there), is
        // read again in one step, so the views a watcher is given follow one another. The changes end
        // once the instance has closed and its final state is read, and so do they.
        const triggers = Stream.merge(
          Stream.map(facade.queries.view.changes(), () => undefined),
          Stream.map(record.changes, () => undefined),
          { haltStrategy: "left" }
        )
        const views = triggers.pipe(
          Stream.mapEffect(() => facade.queries.watched.get(Ref.get(record))),
          Stream.changesWith((previous: View, current: View) => Equal.equals(previous, current)),
          Stream.mapAccumEffect(Option.none<View>(), (previous, current) => Option.match(previous, {
            // The chat exists while it has a record.
            onNone: () => Option.match(current.chat, {
              onNone: () => Effect.fail(new ChatNotFound({ chatId })),
              onSome: (chat) => Effect.succeed([
                Option.some<View>(current),
                [{ _tag: "snapshot", chat, messages: current.messages, live: current.live } as ChatUpdate],
              ] as const),
            }),
            onSome: (last) => Effect.succeed([Option.some<View>(current), [changeBetween(last, current)]] as const),
          })),
          Stream.flattenIterables
        )
        // The view ends when the instance closes, for the reason it closed. A subscription never
        // completes on its own: only the service's terminal control at shutdown ends it, so a removed
        // chat's watch says so once the removal's delete has succeeded, and stays open; a delete that
        // failed fails the watch for the store's reason. A closing service ends the rest itself.
        const ending = Stream.unwrap(Effect.map(closed, (closure): Stream.Stream<ChatUpdate, ChatStorageFailed> => {
          switch (closure._tag) {
            case "Removed":
              return Stream.concat(Stream.as(Stream.fromEffect(removed), { _tag: "removed" } as ChatUpdate), Stream.never)
            case "Failed":
              return Stream.fromEffect(failedWith(closure))
            case "Closed":
              return Stream.never
          }
        }))
        return Stream.concat(views, ending)
      })),
    })
  })
)
