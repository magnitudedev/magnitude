import { Context, Effect, Layer, Option, Stream } from "effect"
import { SessionNotFound, SessionOperationFailed, type SessionError } from "@magnitudedev/protocol"
import type { AgentIntrospection } from "@magnitudedev/agent"
import { AgentRuntime } from "../agent-runtime"
import { AcnActivityTracker } from "../activity-tracker"
import { formatUnknownCause } from "../session-errors"
import { AcnDisplayViewIntrospector } from "./display-views"
import type {
  AcnIntrospectionOverview,
  AcnIntrospectionSession,
  AcnSessionIntrospection,
} from "./types"

export interface AcnIntrospectorApi {
  readonly currentOverview: Effect.Effect<AcnIntrospectionOverview>
  readonly currentSession: (
    sessionId: string,
    forkId?: string | null,
  ) => Effect.Effect<AcnSessionIntrospection, SessionError>
  readonly sessionChanges: (
    sessionId: string,
    forkId?: string | null,
  ) => Stream.Stream<AcnSessionIntrospection, SessionError>
}

export class AcnIntrospector extends Context.Tag("AcnIntrospector")<
  AcnIntrospector,
  AcnIntrospectorApi
>() {}

const runtimeEntryToSession = (entry: {
  readonly sessionId: string
  readonly title: string
  readonly cwd: string
  readonly scratchpadPath: string
  readonly createdAt: number
  readonly updatedAt: number
}): AcnIntrospectionSession => ({
  sessionId: entry.sessionId,
  title: entry.title,
  cwd: entry.cwd,
  scratchpadPath: entry.scratchpadPath,
  createdAt: entry.createdAt,
  updatedAt: entry.updatedAt,
})

const introspectionFailure = (sessionId: string, cause: unknown) =>
  new SessionOperationFailed({
    operation: "AcnIntrospector.currentSession",
    reason: `${sessionId}: ${formatUnknownCause(cause)}`,
  })

export const AcnIntrospectorLive: Layer.Layer<
  AcnIntrospector,
  never,
  AgentRuntime | AcnActivityTracker
> = Layer.effect(
  AcnIntrospector,
  Effect.gen(function* () {
    const runtime = yield* AgentRuntime
    const activity = yield* AcnActivityTracker
    const displayViewIntrospector = yield* Effect.serviceOption(AcnDisplayViewIntrospector)

    const currentDisplayViews = (sessionId: string) =>
      Option.match(displayViewIntrospector, {
        onNone: () => Effect.succeed([]),
        onSome: (introspector) => introspector.current(sessionId),
      })

    const displayViewChanges = (sessionId: string): Stream.Stream<void> =>
      Option.match(displayViewIntrospector, {
        onNone: () => Stream.never as Stream.Stream<void>,
        onSome: (introspector) => introspector.changes(sessionId),
      })

    const currentSessionPayload = (
      session: AcnIntrospectionSession,
      introspection: AgentIntrospection | null,
    ) =>
      Effect.gen(function* () {
        return {
          schemaVersion: 1,
          timestamp: Date.now(),
          session,
          activity: yield* activity.current,
          displayViews: yield* currentDisplayViews(session.sessionId),
          introspection,
        } satisfies AcnSessionIntrospection
      })

    const currentOverview = Effect.gen(function* () {
      const entries = yield* runtime.residentSessions
      return {
        schemaVersion: 1,
        timestamp: Date.now(),
        sessions: entries.map(runtimeEntryToSession),
        activity: yield* activity.current,
      } satisfies AcnIntrospectionOverview
    })

    const sampleSession = Effect.fn("acn.introspector.sample-session")(function* (
      sessionId: string,
      forkId: string | null,
    ) {
      const resident = (yield* runtime.residentSessions).find(
        (candidate) => candidate.sessionId === sessionId,
      )
      if (!resident) return yield* new SessionNotFound({ sessionId })

      // Introspection is ambient. It may join an already-busy generation to
      // obtain a fresh agent snapshot, but it never creates or prolongs one.
      const sampled = yield* runtime.tryWithBusyResident(
        sessionId,
        "introspection-sample",
        (entry) =>
          entry.session.subscribeIntrospection(forkId).pipe(
            Stream.take(1),
            Stream.runHead,
            Effect.map((value) => Option.getOrNull(value)),
            Effect.mapError((cause) => introspectionFailure(sessionId, cause)),
          ),
      )
      return yield* currentSessionPayload(
        runtimeEntryToSession(resident),
        Option.getOrNull(sampled),
      )
    })

    const sessionChanges = (
      sessionId: string,
      forkId: string | null = null,
    ): Stream.Stream<AcnSessionIntrospection, SessionError> =>
      Stream.concat(
        Stream.fromEffect(sampleSession(sessionId, forkId)),
        Stream.merge(runtime.changes, displayViewChanges(sessionId)).pipe(
          Stream.mapEffect(() => sampleSession(sessionId, forkId)),
        ),
      )

    const currentSession = Effect.fn("acn.introspector.current-session")(function* (
      sessionId: string,
      forkId: string | null = null,
    ) {
      return yield* sampleSession(sessionId, forkId)
    })

    return {
      currentOverview,
      currentSession,
      sessionChanges,
    } satisfies AcnIntrospectorApi
  }),
)
