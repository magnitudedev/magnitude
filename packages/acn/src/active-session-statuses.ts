import { Context, Effect, Fiber, Layer, PubSub, Ref, Stream } from "effect"
import type { ActiveSessionStatus, ActiveSessionStatuses, SessionError } from "@magnitudedev/protocol"
import { AgentRuntime } from "./agent-runtime"
import { SessionStore } from "./session-store"
import type { RuntimeEntry } from "./session-types"

export interface ActiveSessionStatusesApi {
  readonly snapshot: Effect.Effect<ActiveSessionStatuses, SessionError>
  readonly stream: Stream.Stream<ActiveSessionStatuses, SessionError>
}

export class ActiveSessionStatusesService extends Context.Tag("ActiveSessionStatuses")<
  ActiveSessionStatusesService,
  ActiveSessionStatusesApi
>() {}

interface TurnSnapshot {
  readonly _tag: string
}

interface AgentStatusSnapshot {
  readonly agents: ReadonlyMap<string, {
    readonly status: string
    readonly parentForkId: string | null
  }>
}

const isRootTurnWorking = (turn: TurnSnapshot): boolean =>
  turn._tag === "active" || turn._tag === "interrupting"

const countWorkingRootWorkers = (agentStatus: AgentStatusSnapshot): number => {
  let count = 0
  for (const agent of agentStatus.agents.values()) {
    if (agent.parentForkId === null && agent.status === "working") count++
  }
  return count
}

const hasWorkingAgent = (agentStatus: AgentStatusSnapshot): boolean => {
  for (const agent of agentStatus.agents.values()) {
    if (agent.status === "working") return true
  }
  return false
}

const sameSnapshot = (left: ActiveSessionStatuses, right: ActiveSessionStatuses): boolean => {
  if (left.sessions.length !== right.sessions.length) return false
  for (let i = 0; i < left.sessions.length; i++) {
    const a = left.sessions[i]
    const b = right.sessions[i]
    if (!a || !b) return false
    if (
      a.sessionId !== b.sessionId ||
      a.workStatus !== b.workStatus ||
      a.activeWorkerCount !== b.activeWorkerCount ||
      a.lastMessageAt !== b.lastMessageAt
    ) return false
  }
  return true
}

export const ActiveSessionStatusesLive: Layer.Layer<
  ActiveSessionStatusesService,
  never,
  AgentRuntime | SessionStore
> =
  Layer.scoped(
    ActiveSessionStatusesService,
    Effect.gen(function* () {
      const runtime = yield* AgentRuntime
      const store = yield* SessionStore
      const changes = yield* PubSub.unbounded<void>()
      const watcherFibers = yield* Ref.make<ReadonlyMap<string, Fiber.RuntimeFiber<void, unknown>>>(new Map())

      const publishChange = PubSub.publish(changes, undefined).pipe(Effect.asVoid)

      const statusForEntry = (entry: RuntimeEntry): Effect.Effect<ActiveSessionStatus | null, SessionError> =>
        Effect.gen(function* () {
          const meta = yield* store.readProtocolMeta(entry.id)
          if (!meta) return null

          const rootTurn = yield* entry.session.state.turn.getFork(null)
          const agentStatus = yield* entry.session.state.agentStatus.get()

          const workStatus = isRootTurnWorking(rootTurn) || hasWorkingAgent(agentStatus)
            ? "working"
            : "idle"

          return {
            sessionId: entry.id,
            workStatus,
            activeWorkerCount: countWorkingRootWorkers(agentStatus),
            lastMessageAt: meta.updatedAt,
          }
        })

      const snapshot: Effect.Effect<ActiveSessionStatuses, SessionError> =
        Effect.gen(function* () {
          const entries = yield* runtime.getAllEntries()
          const statuses = yield* Effect.forEach(entries, statusForEntry, { concurrency: "unbounded" })
          const sessions = statuses.filter((status): status is ActiveSessionStatus => status !== null)
          return {
            sessions: [...sessions].sort((left, right) => left.sessionId.localeCompare(right.sessionId)),
          }
        })

      const watchEntry = (entry: RuntimeEntry) => {
        const rootTurnChanges = entry.session.state.turn.subscribeFork(null).pipe(
          Stream.drop(1),
          Stream.map(() => undefined),
        )
        const agentStatusChanges = entry.session.state.agentStatus.subscribe.pipe(
          Stream.drop(1),
          Stream.map(() => undefined),
        )
        const eventChanges = entry.session.onEvent.pipe(
          Stream.debounce("50 millis"),
          Stream.map(() => undefined),
        )

        return Stream.mergeAll(
          [rootTurnChanges, agentStatusChanges, eventChanges],
          { concurrency: "unbounded" },
        ).pipe(
          Stream.runForEach(() => publishChange),
          Effect.forkScoped,
        )
      }

      const refreshWatchers = Effect.gen(function* () {
        const entries = yield* runtime.getAllEntries()
        const liveById = new Map(entries.map((entry) => [entry.id, entry] as const))
        const current = yield* Ref.get(watcherFibers)
        const next = new Map(current)

        for (const [sessionId, fiber] of current) {
          if (!liveById.has(sessionId)) {
            yield* Fiber.interrupt(fiber)
            next.delete(sessionId)
          }
        }

        for (const entry of entries) {
          if (next.has(entry.id)) continue
          const fiber = yield* watchEntry(entry)
          next.set(entry.id, fiber)
        }

        yield* Ref.set(watcherFibers, next)
      })

      yield* refreshWatchers

      yield* runtime.changes.pipe(
        Stream.runForEach(() => refreshWatchers.pipe(Effect.zipRight(publishChange))),
        Effect.forkScoped,
      )

      const stream = Stream.concat(
        Stream.fromEffect(snapshot),
        Stream.fromPubSub(changes).pipe(Stream.mapEffect(() => snapshot)),
      ).pipe(
        Stream.changesWith(sameSnapshot),
      )

      return {
        snapshot,
        stream,
      }
    }),
  )
