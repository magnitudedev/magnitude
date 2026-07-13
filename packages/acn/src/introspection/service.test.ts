import { describe, expect, it } from "vitest"
import { Effect, Fiber, Layer, Queue, Scope, Stream } from "effect"
import type { AgentIntrospection, CodingAgentSession } from "@magnitudedev/agent"
import type { DisplayViewShape } from "@magnitudedev/protocol"
import { AcnActivityTrackerLive } from "../activity-tracker"
import { AgentRuntime, type AgentRuntimeApi } from "../agent-runtime"
import type { RuntimeEntry } from "../session-types"
import { AcnDisplayViewIntrospector, AcnDisplayViewIntrospectorLive } from "./display-views"
import { AcnIntrospector, AcnIntrospectorLive } from "./service"

const rootShape: DisplayViewShape = {
  timelines: {
    root: {
      kind: "tail",
      limit: 40,
      live: true,
      presentation: "default",
    },
  },
}

const agentIntrospection: AgentIntrospection = {
  timestamp: 1,
  runtime: {
    engineName: "test",
    schemaVersion: "test",
    timestamp: 1,
    projections: [],
  },
  projections: [],
  addressedAtlas: [],
  display: {
    timelines: [],
  },
}

const makeSession = (queue: Queue.Queue<AgentIntrospection>): CodingAgentSession => ({
  on: {
    restoreQueuedMessages: Stream.never,
  },
  state: {
    turn: {
      getFork: () => Effect.die("unused test session turn.getFork"),
      subscribeFork: () => Stream.die("unused test session turn.subscribeFork"),
    },
    agentStatus: {
      get: () => Effect.die("unused test session agentStatus.get"),
      subscribe: Stream.die("unused test session agentStatus.subscribe"),
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
  refreshConfig: () => Effect.void,
  publishInitialTask: () => Effect.void,
  onEvent: Stream.never,
  onError: Stream.never,
    subscribeIntrospection: () => Stream.fromQueue(queue),
})

const makeLayer = (queue: Queue.Queue<AgentIntrospection>) =>
  Layer.unwrapEffect(Effect.gen(function* () {
    const scope = yield* Scope.make()
    const entry = {
      id: "session-a",
      title: "Session A",
      cwd: "/tmp/session-a",
      scratchpadPath: "/tmp/session-a/scratchpad.md",
      createdAt: 1,
      updatedAt: 1,
      session: makeSession(queue),
      scope,
    } satisfies RuntimeEntry

    const runtime: AgentRuntimeApi = {
      getLive: (sessionId) => Effect.succeed(sessionId === entry.id ? entry : null),
      getAllEntries: () => Effect.succeed([entry]),
      getOrStart: () => Effect.succeed(entry),
      requireOrStart: () => Effect.succeed(entry),
      dispose: () => Effect.void,
      touchEntry: () => Effect.void,
      retainEntry: () => Effect.void,
      releaseEntry: () => Effect.void,
      evictIdleSessions: () => Effect.void,
      disposeAll: () => Effect.void,
      hasActiveWork: Effect.succeed(false),
      count: Effect.succeed(1),
      changes: Stream.never,
    }

    const runtimeLayer = Layer.succeed(AgentRuntime, runtime)
    const withActivity = Layer.provideMerge(AcnActivityTrackerLive, runtimeLayer)
    const withDisplay = Layer.provideMerge(AcnDisplayViewIntrospectorLive, withActivity)
    return Layer.provideMerge(AcnIntrospectorLive, withDisplay)
  }))

describe("AcnIntrospector", () => {
  it("emits selected-session introspection when display view state changes", async () => {
    const queue = await Effect.runPromise(Queue.unbounded<AgentIntrospection>())
    const program = Effect.gen(function* () {
      const introspector = yield* AcnIntrospector
      const displayViews = yield* AcnDisplayViewIntrospector

      const fiber = yield* introspector.sessionChanges("session-a").pipe(
        Stream.take(2),
        Stream.runCollect,
        Effect.fork,
      )

      yield* Effect.sleep("10 millis")
      yield* Queue.offer(queue, agentIntrospection)
      yield* Effect.sleep("10 millis")
      yield* displayViews.setShape("session-a", "view-main", rootShape)

      return yield* Fiber.join(fiber)
    })

    const result = await Effect.runPromise(
      program.pipe(Effect.provide(makeLayer(queue)))
    )
    const emissions = [...result]

    expect(emissions).toHaveLength(2)
    expect(emissions[0].displayViews).toHaveLength(0)
    expect(emissions[1].displayViews).toHaveLength(1)
    expect(emissions[1].displayViews[0]).toMatchObject({
      sessionId: "session-a",
      viewId: "view-main",
      shape: rootShape,
    })
    expect(emissions[1].introspection).toEqual(agentIntrospection)
  })
})
