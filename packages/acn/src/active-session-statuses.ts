import { Context, Effect, Layer, Stream } from "effect"
import type {
  ActiveSessionStatus,
  ActiveSessionStatuses,
  SessionError,
} from "@magnitudedev/protocol"
import { AgentRuntime } from "./agent-runtime"
import { SessionStore } from "./session-store"

export interface ActiveSessionStatusesApi {
  readonly snapshot: Effect.Effect<ActiveSessionStatuses, SessionError>
  readonly stream: Stream.Stream<ActiveSessionStatuses, SessionError>
}

export class ActiveSessionStatusesService extends Context.Tag("ActiveSessionStatuses")<
  ActiveSessionStatusesService,
  ActiveSessionStatusesApi
>() {}

const sameSnapshot = (left: ActiveSessionStatuses, right: ActiveSessionStatuses): boolean =>
  left.sessions.length === right.sessions.length &&
  left.sessions.every((value, index) => {
    const other = right.sessions[index]
    return (
      other !== undefined &&
      value.sessionId === other.sessionId &&
      value.workStatus === other.workStatus &&
      value.activeWorkerCount === other.activeWorkerCount &&
      value.lastMessageAt === other.lastMessageAt
    )
  })

export const ActiveSessionStatusesLive: Layer.Layer<
  ActiveSessionStatusesService,
  never,
  AgentRuntime | SessionStore
> = Layer.effect(
  ActiveSessionStatusesService,
  Effect.gen(function* () {
    const runtime = yield* AgentRuntime
    const store = yield* SessionStore

    const snapshot: Effect.Effect<ActiveSessionStatuses, SessionError> = Effect.gen(function* () {
      const statuses = yield* Effect.forEach(
        yield* runtime.residentSessions,
        (resident) =>
          store.readProtocolMeta(resident.sessionId).pipe(
            Effect.map((meta): ActiveSessionStatus | null =>
              meta
                ? {
                    sessionId: resident.sessionId,
                    workStatus: resident.workStatus._tag === "Working" ? "working" : "idle",
                    activeWorkerCount: resident.workStatus.workerCount,
                    lastMessageAt: meta.updatedAt,
                  }
                : null,
            ),
          ),
        { concurrency: "unbounded" },
      )
      return {
        sessions: statuses
          .filter((status): status is ActiveSessionStatus => status !== null)
          .sort((left, right) => left.sessionId.localeCompare(right.sessionId)),
      }
    })

    return {
      snapshot,
      stream: Stream.concat(
        Stream.fromEffect(snapshot),
        runtime.changes.pipe(Stream.mapEffect(() => snapshot)),
      ).pipe(Stream.changesWith(sameSnapshot)),
    }
  }),
)
