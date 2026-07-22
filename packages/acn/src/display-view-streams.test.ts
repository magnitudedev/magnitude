import { describe, expect, it } from "vitest"
import { Effect, Fiber, Layer, Option, PubSub, Queue, Ref, Scope, Stream } from "effect"
import type { AgentLifecycleState, CodingAgentSession, ForkTurnState } from "@magnitudedev/agent"
import { type DisplayState, type DisplayViewShape } from "@magnitudedev/protocol"
import {
  AgentRuntime,
  type AgentRuntimeApi,
  type SessionRetirementObserver,
} from "./agent-runtime"
import { DisplayViewStreams, DisplayViewStreamsLive } from "./display-view-streams"
import { AcnSubscriptionsLive } from "./acn-subscriptions"
import type { RuntimeEntry } from "./session-types"

const rootShape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 100, live: true, presentation: "default" },
  },
}

const compactShape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 20, live: true, presentation: "default" },
  },
}

const displayState = (title: string): DisplayState => ({
  session: { sessionId: "s1", title, cwd: "/tmp" },
  timelines: {},
  agents: {},
  actors: {},
  tasks: {
    byId: {},
    order: [],
    summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 },
  },
})

const idleTurn: ForkTurnState = {
  _tag: "idle",
  completedTurns: 0,
  triggers: [],
  pendingInboundCommunications: [],
  parentForkId: null,
  connectionRetryCount: 0,
}

const idleAgents: AgentLifecycleState = {
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

const makeSession = (
  title: string,
  closed: Ref.Ref<string[]>,
  shapes: Ref.Ref<DisplayViewShape[]>,
): CodingAgentSession => ({
  on: { restoreQueuedMessages: Stream.never },
  state: {
    work: {
      get: () => Effect.succeed({ _tag: "Quiescent" as const, workerCount: 0 }),
      subscribe: Stream.succeed({ _tag: "Quiescent" as const, workerCount: 0 }),
    },
    turn: {
      getFork: () => Effect.succeed(idleTurn),
      subscribeFork: () => Stream.succeed(idleTurn),
    },
    agentStatus: {
      get: () => Effect.succeed(idleAgents),
      subscribe: Stream.succeed(idleAgents),
    },
  },
  displayView: {
    stream: () => Stream.succeed({ shape: rootShape, state: displayState(title) }).pipe(Stream.concat(Stream.never)),
    snapshot: () => Effect.succeed({ shape: rootShape, state: displayState(title) }),
    setShape: (_viewId, shape) => Ref.update(shapes, (all) => [...all, shape]),
    close: (viewId) => Ref.update(closed, (all) => [...all, viewId]),
  },
  send: () => Effect.void,
  interrupt: () => Effect.void,
  publishInitialTask: () => Effect.void,
  onEvent: Stream.never,
  onError: Stream.never,
  subscribeIntrospection: () => Stream.never,
})

const makeSetup = Effect.gen(function* () {
  const closed = yield* Ref.make<string[]>([])
  const shapes = yield* Ref.make<DisplayViewShape[]>([])
  const generation = yield* Ref.make(1)
  const entry = yield* Ref.make<RuntimeEntry | null>(null)
  const busy = yield* Ref.make(false)
  const observers = yield* Ref.make(new Set<SessionRetirementObserver>())
  const changes = yield* PubSub.unbounded<void>()
  const withSessionCalls = yield* Ref.make(0)
  const makeEntry = Effect.fn("test.display-entry")(function* (title: string) {
    const scope = yield* Scope.make()
    return {
      id: "s1",
      createdAt: 1,
      updatedAt: 1,
      title,
      cwd: "/tmp",
      scratchpadPath: "/tmp/scratchpad.md",
      session: makeSession(title, closed, shapes),
      scope,
    } satisfies RuntimeEntry
  })
  yield* Ref.set(entry, yield* makeEntry("generation-1"))

  const runtime: AgentRuntimeApi = {
    withSession: (_sessionId, _label, use) =>
      Effect.gen(function* () {
        yield* Ref.update(withSessionCalls, (count) => count + 1)
        const current = yield* Ref.get(entry)
        if (!current) return yield* Effect.die("missing fake resident")
        yield* Ref.set(busy, true)
        return yield* use(current, yield* Ref.get(generation)).pipe(
          Effect.ensuring(Ref.set(busy, false)),
        )
      }),
    withSessionRequest: () => Effect.die("unused"),
    tryWithResident: (_sessionId, _label, use) =>
      Effect.gen(function* () {
        const current = yield* Ref.get(entry)
        return current
          ? Option.some(yield* use(current, yield* Ref.get(generation)))
          : Option.none()
      }),
    tryWithBusyResident: (_sessionId, _label, use) =>
      Effect.gen(function* () {
        const current = yield* Ref.get(entry)
        if (!current || !(yield* Ref.get(busy))) return Option.none()
        return Option.some(yield* use(current, yield* Ref.get(generation)))
      }),
    residentSessions: Effect.succeed([]),
    dispose: () => Effect.void,
    deleteSession: (_sessionId, remove) => remove,
    registerRetirementObserver: (observer) =>
      Ref.update(observers, (all) => new Set(all).add(observer)).pipe(
        Effect.as(
          Ref.update(observers, (all) => {
            const next = new Set(all)
            next.delete(observer)
            return next
          }),
        ),
      ),
    changes: Stream.fromPubSub(changes),
  }

  const layer = DisplayViewStreamsLive.pipe(
    Layer.provide(Layer.mergeAll(
      Layer.succeed(AgentRuntime, runtime),
      AcnSubscriptionsLive,
    )),
  )
  return {
    layer,
    closed,
    shapes,
    entry,
    generation,
    observers,
    changes,
    withSessionCalls,
    makeEntry,
  }
})

describe("DisplayViewStreams", () => {
  it("keeps a passive outer stream without materializing an idle runtime", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const fiber = yield* streams
          .getDisplayViewStream("s1", "view-a", rootShape)
          .pipe(Stream.runDrain, Effect.fork)
        yield* Effect.yieldNow()
        expect(yield* Ref.get(setup.withSessionCalls)).toBe(0)
        yield* Fiber.interrupt(fiber)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("materializes shape demand and returns the authoritative full state", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const event = yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        return yield* streams.setDisplayViewShape("s1", "view-a", rootShape)
      }).pipe(Effect.provide(setup.layer))
      expect(event._tag).toBe("state")
      expect(event.state.session.title).toBe("generation-1")
      expect(yield* Ref.get(setup.withSessionCalls)).toBe(1)
    })
    await Effect.runPromise(program)
  })

  it("does not let a passive reconnect mutate the desired shape", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        yield* streams.setDisplayViewShape("s1", "view-a", rootShape)
        const reconnect = yield* streams
          .getDisplayViewStream("s1", "view-a", compactShape)
          .pipe(Stream.runDrain, Effect.fork)
        yield* Effect.yieldNow()
        yield* streams.requestDisplayViewSnapshot("s1", "view-a")
        const shapes = yield* Ref.get(setup.shapes)
        expect(shapes.at(-1)).toEqual(rootShape)
        yield* Fiber.interrupt(reconnect)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("detaches on eviction without clearing the outer stream, then reattaches a new generation", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        yield* streams.setDisplayViewShape("s1", "view-a", rootShape)
        const received = yield* Queue.unbounded<string | null>()
        const streamFiber = yield* streams.getDisplayViewStream("s1", "view-a", rootShape).pipe(
          Stream.tap((event) =>
            event._tag === "state"
              ? Queue.offer(received, event.state.session.title)
              : Effect.void,
          ),
          Stream.runDrain,
          Effect.fork,
        )
        expect(yield* Queue.take(received)).toBe("generation-1")

        for (const observer of yield* Ref.get(setup.observers)) {
          yield* observer.retire({ sessionId: "s1", generation: 1 })
        }
        expect(yield* Ref.get(setup.closed)).toEqual([])

        yield* Ref.set(setup.generation, 2)
        yield* Ref.set(setup.entry, yield* setup.makeEntry("generation-2"))
        yield* streams.requestDisplayViewSnapshot("s1", "view-a")
        expect(yield* Queue.take(received)).toBe("generation-2")
        yield* Fiber.interrupt(streamFiber)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("retains a shared registration until the final subscriber leaves", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        yield* streams.setDisplayViewShape("s1", "shared", rootShape)
        const subscribed = yield* Queue.unbounded<void>()
        const observeSubscription = (shape: DisplayViewShape) =>
          streams.getDisplayViewStream("s1", "shared", shape).pipe(
            Stream.tap(() => Queue.offer(subscribed, undefined)),
            Stream.runDrain,
            Effect.fork,
          )
        const first = yield* streams
          .getDisplayViewStream("s1", "shared", rootShape)
          .pipe(
            Stream.tap(() => Queue.offer(subscribed, undefined)),
            Stream.runDrain,
            Effect.fork,
          )
        const second = yield* observeSubscription(rootShape)
        yield* Queue.take(subscribed)
        yield* Queue.take(subscribed)

        yield* Fiber.interrupt(first)
        const whileShared = yield* observeSubscription(compactShape)
        yield* Queue.take(subscribed)
        yield* streams.requestDisplayViewSnapshot("s1", "shared")
        expect((yield* Ref.get(setup.shapes)).at(-1)).toEqual(rootShape)

        yield* Fiber.interrupt(second)
        yield* Fiber.interrupt(whileShared)
        const successor = yield* streams
          .getDisplayViewStream("s1", "shared", compactShape)
          .pipe(Stream.runDrain, Effect.fork)
        yield* Effect.sleep("1 millis")
        yield* streams.requestDisplayViewSnapshot("s1", "shared")
        expect((yield* Ref.get(setup.shapes)).at(-1)).toEqual(compactShape)
        yield* Fiber.interrupt(successor)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })
})
