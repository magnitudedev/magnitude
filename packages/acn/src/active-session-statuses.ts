import { Context, Effect, Layer, Stream } from "effect"
import type { ActiveSessionStatus, ActiveSessionStatuses } from "@magnitudedev/acn-protocol"
import { AgentRuntime } from "./agent-runtime"
import { SessionInspector } from "./session-inspector"

export interface ActiveSessionStatusesApi {
  readonly snapshot: Effect.Effect<ActiveSessionStatuses>
  readonly stream: Stream.Stream<ActiveSessionStatuses>
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
  AgentRuntime | SessionInspector
> = Layer.effect(
  ActiveSessionStatusesService,
  Effect.gen(function* () {
    const runtime = yield* AgentRuntime
    const inspector = yield* SessionInspector

    const snapshot: Effect.Effect<ActiveSessionStatuses> = Effect.gen(function* () {
      const statuses = yield* Effect.forEach(
        yield* runtime.sessionRuntimes,
        (runtime) =>
          inspector.get(runtime.sessionId).pipe(
            Effect.map((meta): ActiveSessionStatus | null => ({
              sessionId: runtime.sessionId,
              workStatus: runtime.workStatus._tag === "Working" ? "working" : "idle",
              activeWorkerCount: runtime.workStatus.workerCount,
              lastMessageAt: meta.updatedAt,
            })),
            Effect.catchTags({
              SessionNotFound: () => Effect.succeed(null),
              SessionMetadataUnreadable: () => Effect.succeed(null),
            }),
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
