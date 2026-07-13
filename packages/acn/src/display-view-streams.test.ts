import { describe, expect, it } from "vitest"
import { Deferred, Duration, Effect, Fiber, Layer, Queue, Scope, Stream } from "effect"
import type { AgentLifecycleState, CodingAgentSession, ForkTurnState } from "@magnitudedev/agent"
import { type DisplayState, type DisplayViewShape } from "@magnitudedev/protocol"
import { AgentRuntime, type AgentRuntimeApi } from "./agent-runtime"
import { DisplayViewStreams, DisplayViewStreamsLive } from "./display-view-streams"
import type { RuntimeEntry } from "./session-types"

const rootShape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 100, live: true, presentation: "default" },
  },
}

const workerShape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 100, live: true, presentation: "default" },
    "worker:one": { kind: "tail", limit: 50, live: true, presentation: "default" },
  },
}

const state: DisplayState = {
  session: { sessionId: "s1", title: null, cwd: "/tmp" },
  timelines: {
    root: {
      mode: "idle",
      messages: { byId: {}, order: [] },
      streamingMessageId: null,
      window: {
        start: 0,
        end: 0,
        totalCount: 0,
        hasMoreBefore: false,
        hasMoreAfter: false,
      },
      presentation: {
        mode: "default",
        entries: [],
        statusSlot: { kind: "none" },
      },
    },
  },
  agents: {},
  actors: {},
  tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
}

const noSessionEvents = {
  restoreQueuedMessages: Stream.empty,
}

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

const makeCodingAgentSession = (displayView: CodingAgentSession["displayView"]): CodingAgentSession => ({
  on: noSessionEvents,
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
  displayView,
  send: () => Effect.void,
  interrupt: () => Effect.void,
  refreshConfig: () => Effect.void,
  publishInitialTask: () => Effect.void,
  onEvent: Stream.never,
  onError: Stream.never,
  subscribeIntrospection: () => Stream.never,
})

const makeRuntimeLayer = (displayView: CodingAgentSession["displayView"]) => {
  return Layer.effect(
    AgentRuntime,
    Effect.gen(function* () {
      const scope = yield* Scope.make()
      const entry: RuntimeEntry = {
        id: "s1",
        createdAt: 1,
        updatedAt: 1,
        title: "Session",
        cwd: "/tmp",
        scratchpadPath: "/tmp/scratchpad.md",
        session: makeCodingAgentSession(displayView),
        scope,
      }
      return {
        getLive: () => Effect.succeed(entry),
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
      } satisfies AgentRuntimeApi
    }),
  )
}

const provideStreams = (displayView: CodingAgentSession["displayView"]) =>
  DisplayViewStreamsLive.pipe(
    Layer.provideMerge(makeRuntimeLayer(displayView)),
  )

describe("DisplayViewStreams", () => {
  it("StreamDisplayView opens the semantic view from its shape", async () => {
    const calls: Array<{ readonly kind: "setShape" | "stream"; readonly viewId: string; readonly shape?: DisplayViewShape }> = []
    const displayView: CodingAgentSession["displayView"] = {
      stream: (viewId) =>
        Stream.fromIterable([{ shape: rootShape, state }]).pipe(
          Stream.tap(() => Effect.sync(() => calls.push({ kind: "stream", viewId })))
        ),
      snapshot: () => Effect.succeed({ shape: rootShape, state }),
      setShape: (viewId, shape) =>
        Effect.sync(() => calls.push({ kind: "setShape", viewId, shape })),
      close: () => Effect.void,
    }

    await Effect.runPromise(Effect.gen(function* () {
      const streams = yield* DisplayViewStreams
      yield* streams.getDisplayViewStream("s1", "view-a", rootShape).pipe(Stream.take(1), Stream.runCollect)
    }).pipe(Effect.provide(provideStreams(displayView))))

    expect(calls).toEqual([
      { kind: "setShape", viewId: "view-a", shape: rootShape },
      { kind: "stream", viewId: "view-a" },
    ])
  })

  it("SetDisplayViewShape opens the semantic view and stream attaches read-only", async () => {
    const calls: Array<{ readonly kind: "setShape" | "stream"; readonly viewId: string; readonly shape?: DisplayViewShape }> = []
    const displayView: CodingAgentSession["displayView"] = {
      stream: (viewId) =>
        Stream.fromIterable([{ shape: rootShape, state }]).pipe(
          Stream.tap(() => Effect.sync(() => calls.push({ kind: "stream", viewId })))
        ),
      snapshot: () => Effect.succeed({ shape: rootShape, state }),
      setShape: (viewId, shape) =>
        Effect.sync(() => calls.push({ kind: "setShape", viewId, shape })),
      close: () => Effect.void,
    }

    await Effect.runPromise(Effect.gen(function* () {
      const streams = yield* DisplayViewStreams
      yield* streams.setDisplayViewShape("s1", "view-a", rootShape)
      yield* streams.getDisplayViewStream("s1", "view-a", rootShape).pipe(Stream.take(1), Stream.runCollect)
    }).pipe(Effect.provide(provideStreams(displayView))))

    expect(calls).toEqual([
      { kind: "setShape", viewId: "view-a", shape: rootShape },
      { kind: "stream", viewId: "view-a" },
    ])
  })

  it("updates shape without reopening the shared stream", async () => {
    const setShapes: DisplayViewShape[] = []
    let streamFactoryCalls = 0
    const displayView: CodingAgentSession["displayView"] = {
      stream: () => {
        streamFactoryCalls += 1
        return Stream.fromIterable([{ shape: rootShape, state }]).pipe(Stream.concat(Stream.never))
      },
      snapshot: () => Effect.succeed({ shape: workerShape, state }),
      setShape: (_viewId, shape) =>
        Effect.sync(() => {
          setShapes.push(shape)
        }),
      close: () => Effect.void,
    }

    await Effect.runPromise(Effect.gen(function* () {
      const streams = yield* DisplayViewStreams
      const events = yield* Queue.unbounded<unknown>()
      yield* streams.setDisplayViewShape("s1", "view-a", rootShape)
      const fiber = yield* streams.getDisplayViewStream("s1", "view-a", rootShape).pipe(
        Stream.tap((event) => Queue.offer(events, event)),
        Stream.runDrain,
        Effect.fork,
      )
      yield* Queue.take(events)
      yield* streams.setDisplayViewShape("s1", "view-a", workerShape)
      yield* Fiber.interrupt(fiber)
    }).pipe(Effect.provide(provideStreams(displayView))))

    expect(streamFactoryCalls).toBe(1)
    expect(setShapes).toEqual([rootShape, workerShape])
  })

  it("stream detach closes the view when refCount reaches 0", async () => {
    const closed: string[] = []
    const displayView: CodingAgentSession["displayView"] = {
      stream: () => Stream.fromIterable([{ shape: rootShape, state }]).pipe(Stream.concat(Stream.never)),
      snapshot: () => Effect.succeed({ shape: rootShape, state }),
      setShape: () => Effect.void,
      close: (viewId) => Effect.sync(() => closed.push(viewId)),
    }

    await Effect.runPromise(Effect.gen(function* () {
      const streams = yield* DisplayViewStreams
      const events = yield* Queue.unbounded<unknown>()
      const drained = yield* Deferred.make<void>()

      yield* streams.setDisplayViewShape("s1", "view-a", rootShape)
      const fiber = yield* streams.getDisplayViewStream("s1", "view-a", rootShape).pipe(
        Stream.tap((event) => Queue.offer(events, event)),
        Stream.runDrain,
        Effect.ensuring(Deferred.succeed(drained, undefined)),
        Effect.fork,
      )

      yield* Queue.take(events)
      yield* Fiber.interrupt(fiber)
      yield* Deferred.await(drained).pipe(
        Effect.timeoutFail({
          duration: Duration.millis(100),
          onTimeout: () => "display stream did not detach",
        })
      )
    }).pipe(Effect.provide(provideStreams(displayView))))

    expect(closed).toEqual(["view-a"])
  })
})
