import { Context, Effect, Layer, Option, Ref, Stream } from "effect"
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
  readonly id: string
  readonly title: string
  readonly cwd: string
  readonly scratchpadPath: string
  readonly createdAt: number
  readonly updatedAt: number
}): AcnIntrospectionSession => ({
  sessionId: entry.id,
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
> =
  Layer.effect(
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
        const entries = yield* runtime.getAllEntries()
        return {
          schemaVersion: 1,
          timestamp: Date.now(),
          sessions: entries.map(runtimeEntryToSession),
          activity: yield* activity.current,
        } satisfies AcnIntrospectionOverview
      })

      const sessionChanges = (
        sessionId: string,
        forkId: string | null = null,
      ): Stream.Stream<AcnSessionIntrospection, SessionError> =>
        Stream.unwrap(
          Effect.gen(function* () {
            const entry = yield* runtime.getLive(sessionId)
            if (!entry) return yield* new SessionNotFound({ sessionId })
            const session = runtimeEntryToSession(entry)
            const latestIntrospection = yield* Ref.make<AgentIntrospection | null>(null)

            const sampleLatest = Ref.get(latestIntrospection).pipe(
              Effect.flatMap((introspection) =>
                introspection
                  ? currentSessionPayload(session, introspection).pipe(Effect.map(Option.some))
                  : Effect.succeed(Option.none<AcnSessionIntrospection>())
              ),
            )

            const agentChanges = entry.session.subscribeIntrospection(forkId).pipe(
              Stream.mapError((cause) => introspectionFailure(sessionId, cause)),
              Stream.mapEffect((introspection) =>
                Ref.set(latestIntrospection, introspection).pipe(
                  Effect.zipRight(currentSessionPayload(session, introspection)),
                )
              ),
            )

            const displayChanges = displayViewChanges(sessionId).pipe(
              Stream.mapEffect(() => sampleLatest),
              Stream.filterMap((payload) => payload),
            )

            const activityChanges = activity.changes.pipe(
              Stream.mapEffect(() => sampleLatest),
              Stream.filterMap((payload) => payload),
            )

            return Stream.mergeAll(
              [agentChanges, displayChanges, activityChanges],
              { concurrency: "unbounded" },
            )
          })
        )

      const currentSession = Effect.fn("acn.introspector.current-session")(function* (
        sessionId: string,
        forkId: string | null = null,
      ) {
        const introspection = yield* sessionChanges(sessionId, forkId).pipe(
          Stream.take(1),
          Stream.runHead,
          Effect.map((option) => Option.getOrNull(option)),
        )
        if (introspection) return introspection
        return yield* introspectionFailure(sessionId, "introspection stream ended before emitting")
      })

      return {
        currentOverview,
        currentSession,
        sessionChanges,
      } satisfies AcnIntrospectorApi
    }),
  )
