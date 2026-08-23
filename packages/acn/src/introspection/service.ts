import { Context, Effect, Layer, Option, Stream } from "effect"
import { SessionNotFound, SessionOperationFailed, type SessionError } from "@magnitudedev/acn-protocol"
import type { AgentIntrospection } from "@magnitudedev/agent"
import { AgentRuntime } from "../agent-runtime"
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

const introspectionFailure = (sessionId: string, cause: unknown) =>
  new SessionOperationFailed({
    operation: "AcnIntrospector.currentSession",
    reason: `${sessionId}: ${formatUnknownCause(cause)}`,
  })

export const AcnIntrospectorLive: Layer.Layer<
  AcnIntrospector,
  never,
  AgentRuntime
> = Layer.effect(
  AcnIntrospector,
  Effect.gen(function* () {
    const runtime = yield* AgentRuntime
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
          schemaVersion: 3,
          timestamp: Date.now(),
          session,
          displayViews: yield* currentDisplayViews(session.sessionId),
          introspection,
        } satisfies AcnSessionIntrospection
      })

    const currentOverview = Effect.gen(function* () {
      const entries = yield* runtime.sessionRuntimes
      return {
        schemaVersion: 3,
        timestamp: Date.now(),
        sessions: entries,
      } satisfies AcnIntrospectionOverview
    })

    const sampleSession = Effect.fn("acn.introspector.sample-session")(function* (
      sessionId: string,
      forkId: string | null,
    ) {
      const runtimeSnapshot = (yield* runtime.sessionRuntimes).find(
        (candidate) => candidate.sessionId === sessionId,
      )
      if (!runtimeSnapshot) return yield* new SessionNotFound({ sessionId })

      // Introspection is ambient. It may join an already-busy generation to
      // obtain a fresh agent snapshot, but it never creates or prolongs one.
      const sampled = yield* Effect.scoped(Effect.gen(function* () {
        const acquired = yield* runtime.tryAcquireActiveSession(
          sessionId,
          "introspection-sample",
        )
        if (Option.isNone(acquired)) return Option.none<AgentIntrospection>()
        return yield* acquired.value.entry.session.subscribeIntrospection(forkId).pipe(
          Stream.take(1),
          Stream.runHead,
          Effect.mapError((cause) => introspectionFailure(sessionId, cause)),
        )
      }))
      return yield* currentSessionPayload(
        runtimeSnapshot,
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
