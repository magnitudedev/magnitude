import { describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Layer, Option, PubSub, Queue, Ref, Scope, Stream } from "effect"
import type { AgentLifecycleState, CodingAgentSession, ForkTurnState } from "@magnitudedev/agent"
import { DirectoryPathSchema, type DisplayState, type DisplayViewShape } from "@magnitudedev/acn-protocol"
import {
  AgentRuntime,
  type AgentRuntimeApi,
  type SessionRetirementObserver,
} from "./agent-runtime"
import { DisplayViewStreams, DisplayViewStreamsLive, displayViewId } from "./display-view-streams"
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
    accumulatedProductiveMs: 0,
    productiveStartedAt: null,
    lastProductiveMs: 0,
    chainStartedAt: null,
    activeChildCount: 0,
    _currentTurn: null,
    _currentChainId: null,
    _isThinking: false,
    _generation: null,
  },
}

const makeSession = (
  title: string,
  closed: Ref.Ref<string[]>,
  shapes: Ref.Ref<ReadonlyArray<readonly [string, DisplayViewShape]>>,
  displayStream: Stream.Stream<{ shape: DisplayViewShape; state: DisplayState }> = Stream.succeed({
    shape: rootShape,
    state: displayState(title),
  }).pipe(Stream.concat(Stream.never)),
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
    stream: () => displayStream,
    snapshot: () => Effect.succeed({ shape: rootShape, state: displayState(title) }),
    setShape: (viewId, shape) => Ref.update(shapes, (all) => [...all, [viewId, shape] as const]),
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
  const shapes = yield* Ref.make<ReadonlyArray<readonly [string, DisplayViewShape]>>([])
  const generation = yield* Ref.make(1)
  const entry = yield* Ref.make<RuntimeEntry | null>(null)
  const busy = yield* Ref.make(false)
  const observers = yield* Ref.make(new Set<SessionRetirementObserver>())
  const changes = yield* PubSub.unbounded<void>()
  const withSessionCalls = yield* Ref.make(0)
  const makeEntry = Effect.fn("test.display-entry")(function* (
    title: string,
    displayStream?: Stream.Stream<{ shape: DisplayViewShape; state: DisplayState }>,
  ) {
    const scope = yield* Scope.make()
    return {
      id: "s1",
      createdAt: 1,
      updatedAt: 1,
      title,
      cwd: DirectoryPathSchema.make("/tmp"),
      scratchpadPath: "/tmp/scratchpad.md",
      session: makeSession(title, closed, shapes, displayStream),
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
    Layer.provide(Layer.succeed(AgentRuntime, runtime)),
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
    busy,
    makeEntry,
  }
})

const firstTitle = (stream: Stream.Stream<{ readonly _tag: string }, unknown>) =>
  stream.pipe(
    Stream.filter((event): event is { readonly _tag: "state"; readonly state: DisplayState } => event._tag === "state"),
    Stream.runHead,
    Effect.map((event) => Option.map(event, (value) => value.state.session.title)),
  )

describe("DisplayViewStreams", () => {
  it("derives one stable view identity per shape", () => {
    expect(displayViewId(rootShape)).toBe(displayViewId({ ...rootShape }))
    expect(displayViewId(rootShape)).not.toBe(displayViewId(compactShape))
  })

  it("materializes the view on subscription and emits the authoritative full state first", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const received = yield* Queue.unbounded<string>()
        const fiber = yield* streams.stream("s1", rootShape).pipe(
          Stream.runForEach((event) => Queue.offer(received, event._tag)),
          Effect.fork,
        )
        expect(yield* Queue.take(received)).toBe("state")
        expect(yield* Ref.get(setup.withSessionCalls)).toBe(1)
        expect((yield* Ref.get(setup.shapes)).at(-1)).toEqual([displayViewId(rootShape), rootShape])
        yield* Fiber.interrupt(fiber)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("rereads a complete snapshot for each new subscriber of a shared view", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const first = yield* streams.stream("s1", rootShape).pipe(Stream.runDrain, Effect.fork)
        yield* Effect.sleep("5 millis")
        const second = yield* firstTitle(streams.stream("s1", rootShape))
        expect(Option.getOrThrow(second)).toBe("generation-1")
        expect(yield* Ref.get(setup.withSessionCalls)).toBe(2)
        yield* Fiber.interrupt(first)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("detaches on eviction without clearing the outer stream, then reopening materializes the new generation", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const received = yield* Queue.unbounded<string | null>()
        const streamFiber = yield* streams.stream("s1", rootShape).pipe(
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
        const reopened = yield* firstTitle(streams.stream("s1", rootShape))
        expect(Option.getOrThrow(reopened)).toBe("generation-2")
        expect(yield* Queue.take(received)).toBe("generation-2")
        yield* Fiber.interrupt(streamFiber)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("does not make retirement wait for an attachment fiber finalizer", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const releaseFinalizer = yield* Deferred.make<void>()
      const delayedDisplay = Stream.succeed({
        shape: rootShape,
        state: displayState("generation-1"),
      }).pipe(
        Stream.concat(Stream.never),
        Stream.ensuring(Deferred.await(releaseFinalizer)),
      )
      yield* Ref.set(setup.entry, yield* setup.makeEntry("generation-1", delayedDisplay))

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const fiber = yield* streams.stream("s1", rootShape).pipe(Stream.runDrain, Effect.fork)
        yield* Effect.sleep("5 millis")
        const observer = [...(yield* Ref.get(setup.observers))][0]
        if (!observer) return yield* Effect.die("missing retirement observer")

        const retired = yield* Effect.raceFirst(
          observer.retire({ sessionId: "s1", generation: 1 }).pipe(Effect.as(true)),
          Effect.sleep("100 millis").pipe(Effect.as(false)),
        )
        yield* Deferred.succeed(releaseFinalizer, undefined)
        expect(retired).toBe(true)
        yield* Fiber.interrupt(fiber)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("retains a shared view until the final subscriber leaves, and keeps shapes apart", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const first = yield* streams.stream("s1", rootShape).pipe(Stream.runDrain, Effect.fork)
        const second = yield* streams.stream("s1", rootShape).pipe(Stream.runDrain, Effect.fork)
        const compact = yield* streams.stream("s1", compactShape).pipe(Stream.runDrain, Effect.fork)
        yield* Effect.sleep("5 millis")
        const materialized = (yield* Ref.get(setup.shapes)).map(([viewId]) => viewId)
        expect(new Set(materialized)).toEqual(new Set([displayViewId(rootShape), displayViewId(compactShape)]))

        // The close runs through the busy-resident path; keep the fake resident busy.
        yield* Ref.set(setup.busy, true)
        yield* Fiber.interrupt(first)
        expect(yield* Ref.get(setup.closed)).toEqual([])
        yield* Fiber.interrupt(second)
        expect(yield* Ref.get(setup.closed)).toEqual([displayViewId(rootShape)])
        yield* Fiber.interrupt(compact)
        expect(yield* Ref.get(setup.closed)).toEqual([displayViewId(rootShape), displayViewId(compactShape)])
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })
})
