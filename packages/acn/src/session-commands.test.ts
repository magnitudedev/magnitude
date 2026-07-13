import { describe, expect, it } from "vitest"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer, Ref, Scope, Stream } from "effect"
import type { AgentLifecycleState, AppEvent, CodingAgentSession, ForkTurnState } from "@magnitudedev/agent"
import { AgentRuntime, type AgentRuntimeApi } from "./agent-runtime"
import { SessionCommands, SessionCommandsLive } from "./session-commands"
import type { RuntimeEntry } from "./session-types"

const idleTurnState: ForkTurnState = {
  _tag: "idle",
  completedTurns: 0,
  triggers: [],
  pendingInboundCommunications: [],
  parentForkId: null,
  connectionRetryCount: 0,
}

const idleAgentStatus: AgentLifecycleState = {
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

const makeSession = (send: CodingAgentSession["send"]): CodingAgentSession => ({
  on: {
    restoreQueuedMessages: Stream.never,
  },
  state: {
    turn: {
      getFork: () => Effect.succeed(idleTurnState),
      subscribeFork: () => Stream.succeed(idleTurnState),
    },
    agentStatus: {
      get: () => Effect.succeed(idleAgentStatus),
      subscribe: Stream.succeed(idleAgentStatus),
    },
  },
  displayView: {
    stream: () => Stream.die("unused test session displayView.stream"),
    snapshot: () => Effect.die("unused test session displayView.snapshot"),
    setShape: () => Effect.die("unused test session displayView.setShape"),
    close: () => Effect.void,
  },
  send,
  interrupt: () => Effect.die("unused test session interrupt"),
  refreshConfig: () => Effect.void,
  publishInitialTask: () => Effect.void,
  onEvent: Stream.never,
  onError: Stream.never,
  subscribeIntrospection: () => Stream.never,
})

const makeEntry = Effect.fn("test.make-session-command-entry")(function* (
  sessionId: string,
  session: CodingAgentSession,
) {
  const scope = yield* Scope.make()
  return {
    id: sessionId,
    createdAt: 1,
    updatedAt: 1,
    title: "Session",
    cwd: process.cwd(),
    scratchpadPath: "/tmp/magnitude-session-commands-scratchpad",
    session,
    scope,
  } satisfies RuntimeEntry
})

const makeLayer = (runtime: AgentRuntimeApi) =>
  SessionCommandsLive.pipe(
    Layer.provide(Layer.mergeAll(
      Layer.succeed(AgentRuntime, runtime),
      BunFileSystem.layer,
      BunPath.layer,
    )),
  )

describe("SessionCommands", () => {
  it("sendUserMessage starts an evicted runtime before sending", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const sentEvents = yield* Ref.make<ReadonlyArray<AppEvent>>([])
      const requireOrStartCalls = yield* Ref.make(0)
      const entry = yield* makeEntry(
        "session-a",
        makeSession((event) => Ref.update(sentEvents, (events) => [...events, event])),
      )

      const runtime: AgentRuntimeApi = {
        getLive: () => Effect.succeed(null),
        getAllEntries: () => Effect.succeed([]),
        getOrStart: () => Effect.die("unused"),
        requireOrStart: (sessionId) =>
          Ref.update(requireOrStartCalls, (count) => count + 1).pipe(
            Effect.as({ ...entry, id: sessionId }),
          ),
        dispose: () => Effect.void,
        disposeAll: () => Effect.void,
        touchEntry: () => Effect.void,
        retainEntry: () => Effect.void,
        releaseEntry: () => Effect.void,
        evictIdleSessions: () => Effect.void,
        hasActiveWork: Effect.succeed(false),
        count: Effect.succeed(0),
        changes: Stream.never,
      }

      yield* Effect.gen(function* () {
        const commands = yield* SessionCommands
        yield* commands.sendUserMessage({
          sessionId: "session-a",
          content: "hello after eviction",
          taskMode: false,
          attachments: [],
        })
      }).pipe(Effect.provide(makeLayer(runtime)))

      const calls = yield* Ref.get(requireOrStartCalls)
      const events = yield* Ref.get(sentEvents)

      expect(calls).toBe(1)
      expect(events).toHaveLength(1)
      expect(events[0]?.type).toBe("user_message")
    }))
  })
})
