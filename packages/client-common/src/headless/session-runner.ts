import { RpcClient } from "@effect/rpc"
import {
  Context,
  Data,
  Effect,
  Layer,
  Option,
  Schema,
  Stream,
} from "effect"
import {
  HeadlessRpcs,
  forkIdToKey,
  type CreateSessionResult,
  type DisplayViewShape,
  type DisplayViewSnapshot,
  type DisplayViewStateEvent,
  type SessionMetadata,
  type SessionOptions,
  type StreamEvent,
} from "@magnitudedev/sdk"
import { applyStreamEvent } from "../sync/apply-stream-event"
import { createDisplayViewStore } from "../sync/display-view-store"

const HEADLESS_TIMELINE_LIMIT = 10_000

export const HeadlessSessionIdSchema = Schema.String.pipe(Schema.brand("HeadlessSessionId"))
export type HeadlessSessionId = Schema.Schema.Type<typeof HeadlessSessionIdSchema>
export const HeadlessDisplayViewIdSchema = Schema.String.pipe(Schema.brand("HeadlessDisplayViewId"))
export type HeadlessDisplayViewId = Schema.Schema.Type<typeof HeadlessDisplayViewIdSchema>

export interface HeadlessInitialWork {
  readonly type: "message" | "goal"
  readonly content: string
}

export interface RunHeadlessSessionRequest {
  readonly sessionId: HeadlessSessionId
  readonly cwd: string
  readonly initial: HeadlessInitialWork
  readonly options: SessionOptions
}

export interface HeadlessSessionObserver<ObserverError = never> {
  readonly onSessionCreated?: (sessionId: HeadlessSessionId) => Effect.Effect<void, ObserverError>
  readonly onSnapshot: (snapshot: DisplayViewSnapshot) => Effect.Effect<void, ObserverError>
}

export interface HeadlessSessionResult {
  readonly sessionId: HeadlessSessionId
  readonly session: SessionMetadata
  readonly status: "completed" | "failed" | "interrupted"
  readonly elapsedMs: number
}

export class HeadlessSessionClientFailure extends Data.TaggedError("HeadlessSessionClientFailure")<{
  readonly operation: string
  readonly message: string
}> {}

export class HeadlessSessionStartFailed extends Data.TaggedError("HeadlessSessionStartFailed")<{
  readonly message: string
}> {}

export class HeadlessSessionPersistenceFailed extends Data.TaggedError("HeadlessSessionPersistenceFailed")<{
  readonly sessionId: HeadlessSessionId
  readonly message: string
}> {}

export class HeadlessSessionStreamEnded extends Data.TaggedError("HeadlessSessionStreamEnded")<{
  readonly sessionId: HeadlessSessionId
  readonly message: string
}> {}

export interface HeadlessSessionClient {
  readonly createSession: (request: RunHeadlessSessionRequest) => Effect.Effect<
    CreateSessionResult,
    HeadlessSessionClientFailure
  >
  readonly getSession: (sessionId: HeadlessSessionId) => Effect.Effect<
    SessionMetadata,
    HeadlessSessionClientFailure
  >
  readonly resyncDisplayView: (
    sessionId: HeadlessSessionId,
    viewId: HeadlessDisplayViewId,
  ) => Effect.Effect<DisplayViewStateEvent, HeadlessSessionClientFailure>
  readonly streamDisplayView: (
    sessionId: HeadlessSessionId,
    viewId: HeadlessDisplayViewId,
    shape: DisplayViewShape,
  ) => Stream.Stream<StreamEvent, HeadlessSessionClientFailure>
}

export const HeadlessSessionClient = Context.GenericTag<HeadlessSessionClient>(
  "@magnitudedev/client-common/HeadlessSessionClient",
)

export function makeHeadlessSessionClientLayer(
  protocolLayer: Layer.Layer<RpcClient.Protocol>,
): Layer.Layer<HeadlessSessionClient> {
  const client = RpcClient.make(HeadlessRpcs).pipe(
    Effect.map((rpc): HeadlessSessionClient => ({
      createSession: (request) => rpc.CreateSession({
        cwd: request.cwd,
        draftOwnerId: Option.none(),
        sessionId: Option.some(request.sessionId),
        options: Option.some(request.options),
        initial: Option.some(request.initial.type === "message"
          ? {
              _tag: "message" as const,
              messageId: Option.none(),
              content: request.initial.content,
              visibleMessage: Option.none(),
              taskMode: false,
              imageAttachments: [],
              mentions: [],
            }
          : {
              _tag: "goal" as const,
              objective: request.initial.content,
            }),
      }).pipe(mapClientFailure("CreateSession")),
      getSession: (sessionId) => rpc.GetSession({ sessionId }).pipe(mapClientFailure("GetSession")),
      resyncDisplayView: (sessionId, viewId) => rpc.ResyncDisplayView({
        sessionId,
        viewId,
      }).pipe(mapClientFailure("ResyncDisplayView")),
      streamDisplayView: (sessionId, viewId, shape) => rpc.StreamDisplayView({
        sessionId,
        viewId,
        shape,
        materialize: true,
      }).pipe(Stream.mapError((error) => clientFailure("StreamDisplayView", error))),
    })),
  )

  return Layer.scoped(HeadlessSessionClient, client).pipe(Layer.provide(protocolLayer))
}

export function runHeadlessSession<ObserverError = never>(
  request: RunHeadlessSessionRequest,
  observer: HeadlessSessionObserver<ObserverError>,
): Effect.Effect<
  HeadlessSessionResult,
  | HeadlessSessionClientFailure
  | HeadlessSessionPersistenceFailed
  | HeadlessSessionStartFailed
  | HeadlessSessionStreamEnded
  | ObserverError,
  HeadlessSessionClient
> {
  return Effect.gen(function* () {
    const startedAt = yield* Effect.clockWith((clock) => clock.currentTimeMillis)
    const client = yield* HeadlessSessionClient
    const creation = yield* client.createSession(request)
    const sessionId = yield* sessionIdFromCreation(creation)
    if (sessionId !== request.sessionId) {
      return yield* new HeadlessSessionStartFailed({
        message: `daemon created unexpected session ${sessionId} instead of ${request.sessionId}`,
      })
    }
    if (observer.onSessionCreated) yield* observer.onSessionCreated(sessionId)
    const viewId = HeadlessDisplayViewIdSchema.make(`headless:${sessionId}`)
    const shape = headlessDisplayShape()
    let store: ReturnType<typeof createDisplayViewStore> | null = null

    const last = yield* client.streamDisplayView(sessionId, viewId, shape).pipe(
      Stream.mapEffect((event) => Effect.gen(function* () {
        if (store === null) {
          const initial = event._tag === "state"
            ? event
            : yield* client.resyncDisplayView(sessionId, viewId)
          store = createDisplayViewStore(initial.state, initial.shape)
          return store.acceptedSnapshot()
        }
        let resyncRequested = false
        yield* applyStreamEvent(
          store,
          event,
          () => { resyncRequested = true },
          sessionId,
          viewId,
        )
        if (resyncRequested) {
          const resynced = yield* client.resyncDisplayView(sessionId, viewId)
          store.accept({ shape: resynced.shape, state: resynced.state })
        }
        return store.acceptedSnapshot()
      })),
      Stream.tap(observer.onSnapshot),
      Stream.takeUntil((snapshot) => classifyTerminalStatus(snapshot) !== null),
      Stream.runLast,
    )

    if (Option.isNone(last)) {
      return yield* new HeadlessSessionStreamEnded({
        sessionId,
        message: "Display subscription ended before an initial state was observed",
      })
    }
    const latest = last.value
    const status = classifyTerminalStatus(latest)
    if (status === null) {
      return yield* new HeadlessSessionStreamEnded({
        sessionId,
        message: "Display subscription ended before a terminal session state was observed",
      })
    }

    const session = yield* client.getSession(sessionId)
    const finishedAt = yield* Effect.clockWith((clock) => clock.currentTimeMillis)
    return { sessionId, session, status, elapsedMs: Math.max(0, finishedAt - startedAt) }
  })
}

function sessionIdFromCreation(creation: CreateSessionResult): Effect.Effect<
  HeadlessSessionId,
  HeadlessSessionPersistenceFailed | HeadlessSessionStartFailed
> {
  switch (creation._tag) {
    case "created":
      return Effect.succeed(HeadlessSessionIdSchema.make(creation.metadata.sessionId))
    case "created_message_failed":
      return new HeadlessSessionPersistenceFailed({
        sessionId: HeadlessSessionIdSchema.make(creation.sessionId),
        message: `Session ${creation.sessionId} accepted the initial work but was not durably promoted: ${creation.error}`,
      })
    case "failed":
      return new HeadlessSessionStartFailed({
        message: `Could not create the headless session: ${creation.error}`,
      })
  }
}

function headlessDisplayShape(): DisplayViewShape {
  return {
    timelines: {
      [forkIdToKey(null)]: {
        kind: "tail",
        limit: HEADLESS_TIMELINE_LIMIT,
        live: true,
        presentation: "default",
      },
    },
  }
}

function classifyTerminalStatus(
  snapshot: DisplayViewSnapshot,
): HeadlessSessionResult["status"] | null {
  const root = snapshot.state.actors[forkIdToKey(null)]
  if (!root || root.kind !== "root") return null
  if (root.status._tag !== "Interrupted" && root.status._tag !== "Worked") return null
  const outcome = terminalDisplayOutcome(snapshot)
  if (outcome === null) return null
  if (root.status._tag === "Interrupted") return "interrupted"
  return outcome === "none" ? null : outcome
}

type TerminalDisplayOutcome = "none" | "completed" | "failed"

function terminalDisplayOutcome(snapshot: DisplayViewSnapshot): TerminalDisplayOutcome | null {
  const timeline = snapshot.state.timelines[forkIdToKey(null)]
  if (!timeline || timeline.mode !== "idle" || timeline.streamingMessageId !== null) return null

  const orderedMessageIds = new Set(timeline.messages.order)
  if (orderedMessageIds.size !== timeline.messages.order.length) return null
  const presentationEntries = new Map<string, Extract<
    (typeof timeline.presentation.entries)[number],
    { readonly kind: "message" }
  >>()
  for (const entry of timeline.presentation.entries) {
    if (entry.kind !== "message") continue
    if (!orderedMessageIds.has(entry.messageId) || entry.streaming || presentationEntries.has(entry.messageId)) return null
    const message = timeline.messages.byId[entry.messageId]
    if (!message || message.id !== entry.messageId) return null
    presentationEntries.set(entry.messageId, entry)
  }

  let latestFailureIndex = -1
  let latestSuccessIndex = -1
  const seen = new Set<string>()
  for (const [index, messageId] of timeline.messages.order.entries()) {
    if (seen.has(messageId)) return null
    seen.add(messageId)
    const message = timeline.messages.byId[messageId]
    if (!message || message.id !== messageId) return null
    if (message.type === "error") latestFailureIndex = index
    if (message.type === "assistant_message" && message.content.trim().length > 0) {
      const presentation = presentationEntries.get(messageId)
      if (!presentation) return null
      latestSuccessIndex = index
    }
    if (message.type === "goal_status" && message.status === "finished") {
      latestSuccessIndex = index
    }
  }
  if (latestFailureIndex === -1 && latestSuccessIndex === -1) return "none"
  return latestFailureIndex > latestSuccessIndex ? "failed" : "completed"
}

function mapClientFailure(operation: string) {
  return <A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<A, HeadlessSessionClientFailure, R> =>
    Effect.mapError(effect, (error) => clientFailure(operation, error))
}

function clientFailure(operation: string, error: unknown): HeadlessSessionClientFailure {
  return new HeadlessSessionClientFailure({
    operation,
    message: error instanceof Error ? error.message : String(error),
  })
}
