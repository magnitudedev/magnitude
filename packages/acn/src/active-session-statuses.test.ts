import { describe, expect, it } from "vitest"
import { Effect, Fiber, Layer, Queue, Ref, Scope, Stream } from "effect"
import type { AgentLifecycleState, CodingAgentSession, ForkTurnState } from "@magnitudedev/agent"
import type { SessionMetadata } from "@magnitudedev/protocol"
import { AgentRuntime, type AgentRuntimeApi } from "./agent-runtime"
import { ActiveSessionStatusesLive, ActiveSessionStatusesService } from "./active-session-statuses"
import { SessionStore, type SessionStoreApi } from "./session-store"
import type { RuntimeEntry } from "./session-types"

const idleTurn: ForkTurnState = {
  _tag: "idle",
  completedTurns: 0,
  triggers: [],
  pendingInboundCommunications: [],
  parentForkId: null,
  connectionRetryCount: 0,
}

const activeTurn: ForkTurnState = {
  ...idleTurn,
  _tag: "active",
  turnId: "turn-a",
  chainId: "chain-a",
  toolCalls: [],
  triggeredByUser: true,
  requiresAdvisor: false,
}

const noAgents: AgentLifecycleState = {
  agents: new Map(),
  agentByForkId: new Map(),
  rootWork: {
    phase: "idle",
    chainStartedAt: null,
    lastChainMs: 0,
    activity: null,
    activeChildCount: 0,
    _currentTurnId: null,
    _thinkingCharCount: null,
    _activeToolKey: null,
  },
}

const protocolMeta = (
  sessionId: string,
  updatedAt: number,
): SessionMetadata => ({
  sessionId,
  title: "Session",
  cwd: "/tmp",
  createdAt: 1,
  updatedAt,
  messageCount: 0,
  lastMessage: null,
})

const makeSession = (input: {
  readonly turn: Ref.Ref<ForkTurnState>
  readonly agentStatus: Ref.Ref<AgentLifecycleState>
  readonly turnUpdates: Queue.Queue<ForkTurnState>
  readonly agentStatusUpdates: Queue.Queue<AgentLifecycleState>
}): CodingAgentSession => ({
  on: {
    restoreQueuedMessages: Stream.never,
  },
  state: {
    turn: {
      getFork: () => Ref.get(input.turn),
      subscribeFork: () =>
        Stream.concat(
          Stream.fromEffect(Ref.get(input.turn)),
          Stream.fromQueue(input.turnUpdates),
        ),
    },
    agentStatus: {
      get: () => Ref.get(input.agentStatus),
      subscribe: Stream.concat(
        Stream.fromEffect(Ref.get(input.agentStatus)),
        Stream.fromQueue(input.agentStatusUpdates),
      ),
    },
  },
  displayView: {
    stream: () => Stream.die("unused test session displayView.stream"),
    snapshot: () => Effect.die("unused test session displayView.snapshot"),
    setShape: () => Effect.die("unused test session displayView.setShape"),
    close: () => Effect.void,
  },
  send: () => Effect.die("unused test session send"),
  interrupt: () => Effect.die("unused test session interrupt"),
  publishInitialTask: () => Effect.void,
  onEvent: Stream.never,
  onError: Stream.never,
  subscribeIntrospection: () => Stream.never,
})

const makeEntry = Effect.fn("test.make-runtime-entry")(function* (
  sessionId: string,
  updatedAt: number,
  session: CodingAgentSession,
) {
  const scope = yield* Scope.make()
  return {
    id: sessionId,
    title: "Session",
    cwd: "/tmp",
    scratchpadPath: `/tmp/${sessionId}/scratchpad`,
    createdAt: 1,
    updatedAt,
    session,
    scope,
  } satisfies RuntimeEntry
})

const makeLayer = Effect.gen(function* () {
  const entries = yield* Ref.make<ReadonlyArray<RuntimeEntry>>([])
  const metas = yield* Ref.make(new Map<string, SessionMetadata>())
  const runtimeChanges = yield* Queue.unbounded<void>()

  const runtime: AgentRuntimeApi = {
    getLive: (sessionId) =>
      Ref.get(entries).pipe(Effect.map((all) => all.find((entry) => entry.id === sessionId) ?? null)),
    getAllEntries: () => Ref.get(entries),
    getOrStart: () => Effect.die("unused"),
    requireOrStart: () => Effect.die("unused"),
    dispose: () => Effect.void,
    touchEntry: () => Effect.void,
    retainEntry: () => Effect.void,
    releaseEntry: () => Effect.void,
    evictIdleSessions: () => Effect.void,
    disposeAll: () => Effect.void,
    hasActiveWork: Effect.succeed(false),
    count: Ref.get(entries).pipe(Effect.map((all) => all.length)),
    changes: Stream.fromQueue(runtimeChanges),
  }

  const store: SessionStoreApi = {
    createId: Effect.die("unused"),
    readMeta: () => Effect.die("unused"),
    readProtocolMeta: (sessionId) =>
      Ref.get(metas).pipe(Effect.map((all) => all.get(sessionId) ?? null)),
    promoteDraft: () => Effect.die("unused"),
    listDraftSessionIds: () => Effect.die("unused"),
    listProtocolMetas: () => Effect.die("unused"),
    listSessionCwds: () => Effect.die("unused"),
    deleteSessionFiles: () => Effect.die("unused"),
    validateCwd: () => Effect.die("unused"),
    getScratchpadPath: () => Effect.die("unused"),
    getExecutionContext: () => Effect.die("unused"),

  }

  const layer = ActiveSessionStatusesLive.pipe(
    Layer.provide(Layer.mergeAll(
      Layer.succeed(AgentRuntime, runtime),
      Layer.succeed(SessionStore, store),
    )),
  )

  return { layer, refs: { entries, metas, runtimeChanges } }
})

describe("ActiveSessionStatuses", () => {
  it("reports active root work with zero workers", async () => {
    const turn = await Effect.runPromise(Ref.make<ForkTurnState>(activeTurn))
    const agentStatus = await Effect.runPromise(Ref.make<AgentLifecycleState>(noAgents))
    const turnUpdates = await Effect.runPromise(Queue.unbounded<ForkTurnState>())
    const agentStatusUpdates = await Effect.runPromise(Queue.unbounded<AgentLifecycleState>())
    const session = makeSession({ turn, agentStatus, turnUpdates, agentStatusUpdates })
    const setup = await Effect.runPromise(makeLayer)

    const entry = await Effect.runPromise(makeEntry("session-a", 10, session))
    await Effect.runPromise(Ref.set(setup.refs.entries, [entry]))
    await Effect.runPromise(Ref.update(setup.refs.metas, (map) => new Map(map).set("session-a", protocolMeta("session-a", 42))))

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const statuses = yield* ActiveSessionStatusesService
        return yield* statuses.snapshot
      }).pipe(Effect.provide(setup.layer)),
    )

    expect(result).toEqual({
      sessions: [{
        sessionId: "session-a",
        workStatus: "working",
        activeWorkerCount: 0,
        lastMessageAt: 42,
      }],
    })
  })

  it("streams worker count changes", async () => {
    const turn = await Effect.runPromise(Ref.make<ForkTurnState>(idleTurn))
    const agentStatus = await Effect.runPromise(Ref.make<AgentLifecycleState>(noAgents))
    const turnUpdates = await Effect.runPromise(Queue.unbounded<ForkTurnState>())
    const agentStatusUpdates = await Effect.runPromise(Queue.unbounded<AgentLifecycleState>())
    const session = makeSession({ turn, agentStatus, turnUpdates, agentStatusUpdates })
    const setup = await Effect.runPromise(makeLayer)

    const entry = await Effect.runPromise(makeEntry("session-a", 10, session))
    await Effect.runPromise(Ref.set(setup.refs.entries, [entry]))
    await Effect.runPromise(Ref.update(setup.refs.metas, (map) => new Map(map).set("session-a", protocolMeta("session-a", 10))))

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const statuses = yield* ActiveSessionStatusesService
        const fiber = yield* statuses.stream.pipe(
          Stream.take(2),
          Stream.runCollect,
          Effect.fork,
        )

        yield* Effect.sleep("10 millis")
        const nextStatus: AgentLifecycleState = {
          ...noAgents,
          agents: new Map([
            ["agent-a", {
              agentId: "agent-a",
              forkId: "fork-a",
              parentForkId: null,
              name: "Worker A",
              role: "engineer",
              context: "",
              mode: "spawn",
              taskId: "task-a",
              message: null,
              status: "working",
              lastIdleReason: null,
            }],
          ]),
          agentByForkId: new Map([["fork-a", "agent-a"]]),
        }
        yield* Ref.set(agentStatus, nextStatus)
        yield* Queue.offer(agentStatusUpdates, nextStatus)

        return yield* Fiber.join(fiber)
      }).pipe(Effect.provide(setup.layer)),
    )

    expect([...result]).toEqual([
      {
        sessions: [{
          sessionId: "session-a",
          workStatus: "idle",
          activeWorkerCount: 0,
          lastMessageAt: 10,
        }],
      },
      {
        sessions: [{
          sessionId: "session-a",
          workStatus: "working",
          activeWorkerCount: 1,
          lastMessageAt: 10,
        }],
      },
    ])
  })

  it("removes disposed runtime sessions from snapshots", async () => {
    const turn = await Effect.runPromise(Ref.make<ForkTurnState>(idleTurn))
    const agentStatus = await Effect.runPromise(Ref.make<AgentLifecycleState>(noAgents))
    const turnUpdates = await Effect.runPromise(Queue.unbounded<ForkTurnState>())
    const agentStatusUpdates = await Effect.runPromise(Queue.unbounded<AgentLifecycleState>())
    const session = makeSession({ turn, agentStatus, turnUpdates, agentStatusUpdates })
    const setup = await Effect.runPromise(makeLayer)

    const entry = await Effect.runPromise(makeEntry("session-a", 10, session))
    await Effect.runPromise(Ref.set(setup.refs.entries, [entry]))
    await Effect.runPromise(Ref.update(setup.refs.metas, (map) => new Map(map).set("session-a", protocolMeta("session-a", 10))))

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const statuses = yield* ActiveSessionStatusesService
        const fiber = yield* statuses.stream.pipe(
          Stream.take(2),
          Stream.runCollect,
          Effect.fork,
        )

        yield* Effect.sleep("10 millis")
        yield* Ref.set(setup.refs.entries, [])
        yield* Queue.offer(setup.refs.runtimeChanges, undefined)

        return yield* Fiber.join(fiber)
      }).pipe(Effect.provide(setup.layer)),
    )

    expect([...result]).toEqual([
      {
        sessions: [{
          sessionId: "session-a",
          workStatus: "idle",
          activeWorkerCount: 0,
          lastMessageAt: 10,
        }],
      },
      { sessions: [] },
    ])
  })
})
