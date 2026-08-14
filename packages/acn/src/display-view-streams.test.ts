import { describe, expect, it } from "vitest"
import { Context, Deferred, Effect, Exit, Fiber, Layer, Option, PubSub, Queue, Ref, Scope, Stream } from "effect"
import {
  DisplayViewRuntimeError,
  type AgentLifecycleState,
  type CodingAgentSession,
  type ForkTurnState,
} from "@magnitudedev/agent"
import { Addressed } from "@magnitudedev/event-core"
import {
  type DisplayState,
  type DisplayViewShape,
  type StreamEvent,
} from "@magnitudedev/acn-protocol"
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

const tinyShape: DisplayViewShape = {
  timelines: {
    root: { kind: "tail", limit: 5, live: true, presentation: "default" },
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

type TestGate = {
  readonly entered: Deferred.Deferred<void>
  readonly release: Deferred.Deferred<void>
}

const makeSession = (
  title: string,
  closed: Ref.Ref<string[]>,
  shapes: Ref.Ref<DisplayViewShape[]>,
  closeGate: Ref.Ref<TestGate | null>,
  displayStream: Stream.Stream<
    { shape: DisplayViewShape; state: DisplayState },
    DisplayViewRuntimeError
  > = Stream.succeed({
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
    snapshot: () => Ref.get(shapes).pipe(Effect.map((all) => ({
      shape: all.at(-1) ?? rootShape,
      state: displayState(title),
    }))),
    setShape: (_viewId, shape) => Ref.update(shapes, (all) => [...all, shape]),
    setShapeAndSnapshot: (_viewId, shape) => Ref.update(
      shapes,
      (all) => [...all, shape],
    ).pipe(Effect.as({ shape, state: displayState(title) })),
    close: (viewId) => Effect.gen(function* () {
      const gate = yield* Ref.get(closeGate)
      if (gate !== null) {
        yield* Deferred.succeed(gate.entered, undefined)
        yield* Deferred.await(gate.release).pipe(Effect.timeoutOption("2 seconds"))
      }
      yield* Ref.update(closed, (all) => [...all, viewId])
    }),
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
  const closeGate = yield* Ref.make<TestGate | null>(null)
  const withSessionGate = yield* Ref.make<TestGate | null>(null)
  const busyAttachGate = yield* Ref.make<TestGate | null>(null)
  const busyCloseGate = yield* Ref.make<TestGate | null>(null)
  const makeEntry = Effect.fn("test.display-entry")(function* (
    title: string,
    displayStream?: Stream.Stream<
      { shape: DisplayViewShape; state: DisplayState },
      DisplayViewRuntimeError
    >,
  ) {
    const scope = yield* Scope.make()
    return {
      id: "s1",
      createdAt: 1,
      updatedAt: 1,
      title,
      cwd: "/tmp",
      scratchpadPath: "/tmp/scratchpad.md",
      session: makeSession(title, closed, shapes, closeGate, displayStream),
      scope,
    } satisfies RuntimeEntry
  })
  yield* Ref.set(entry, yield* makeEntry("generation-1"))

  const runtime: AgentRuntimeApi = {
    withSession: (_sessionId, _label, use) =>
      Effect.gen(function* () {
        yield* Ref.update(withSessionCalls, (count) => count + 1)
        const gate = yield* Ref.get(withSessionGate)
        if (gate !== null) {
          yield* Deferred.succeed(gate.entered, undefined)
          yield* Deferred.await(gate.release)
        }
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
    tryWithBusyResident: (_sessionId, label, use) =>
      Effect.gen(function* () {
        const current = yield* Ref.get(entry)
        if (!current || !(yield* Ref.get(busy))) return Option.none()
        const gate = label.startsWith("display-attach:")
          ? yield* Ref.get(busyAttachGate)
          : label.startsWith("display-close:")
            ? yield* Ref.get(busyCloseGate)
            : null
        if (gate !== null) {
          yield* Deferred.succeed(gate.entered, undefined)
          yield* Deferred.await(gate.release)
        }
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
    busy,
    generation,
    observers,
    changes,
    withSessionCalls,
    closeGate,
    withSessionGate,
    busyAttachGate,
    busyCloseGate,
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

  it("materializes state inside subscriber admission when requested", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const event = yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        return yield* streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
          Stream.runHead,
        )
      }).pipe(Effect.provide(setup.layer))

      expect(Option.isSome(event)).toBe(true)
      if (Option.isSome(event)) {
        expect(event.value._tag).toBe("state")
      }
      expect(yield* Ref.get(setup.withSessionCalls)).toBe(1)
    })
    await Effect.runPromise(program)
  })

  it("emits the admission materialization before a cached attachment", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const event = yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        yield* streams.setDisplayViewShape("s1", "headless", rootShape)
        yield* Ref.set(setup.entry, yield* setup.makeEntry("Fresh"))
        yield* Ref.set(setup.generation, 2)
        return yield* streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
          Stream.runHead,
        )
      }).pipe(Effect.provide(setup.layer))

      expect(Option.isSome(event)).toBe(true)
      if (Option.isSome(event) && event.value._tag === "state") {
        expect(event.value.state.session.title).toBe("Fresh")
      }
    })
    await Effect.runPromise(program)
  })

  it("fences stale attachment events before a materializing subscriber's first state", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const updates = yield* PubSub.unbounded<{ shape: DisplayViewShape; state: DisplayState }>()
      yield* Ref.set(setup.entry, yield* setup.makeEntry(
        "generation-1",
        Stream.fromPubSub(updates),
      ))

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const received = yield* Queue.unbounded<StreamEvent>()
        const first = yield* streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
          Stream.tap((event) => Queue.offer(received, event)),
          Stream.runDrain,
          Effect.fork,
        )
        yield* Queue.take(received)
        let warmed = false
        for (let attempt = 0; attempt < 20 && !warmed; attempt += 1) {
          yield* PubSub.publish(updates, { shape: rootShape, state: displayState("warmup") })
          warmed = Option.isSome(yield* Queue.take(received).pipe(Effect.timeoutOption("10 millis")))
        }
        expect(warmed).toBe(true)

        const entered = yield* Deferred.make<void>()
        const release = yield* Deferred.make<void>()
        yield* Ref.set(setup.withSessionGate, { entered, release })
        const second = yield* streams
          .getDisplayViewStream("s1", "headless", compactShape, true)
          .pipe(Stream.runHead, Effect.fork)
        yield* Deferred.await(entered)
        yield* PubSub.publish(updates, { shape: rootShape, state: displayState("stale") })
        yield* Queue.take(received).pipe(Effect.timeoutOption("20 millis"))
        yield* Deferred.succeed(release, undefined)

        const firstForSecond = yield* Fiber.join(second)
        expect(Option.isSome(firstForSecond)).toBe(true)
        if (Option.isSome(firstForSecond) && firstForSecond.value._tag === "state") {
          expect(firstForSecond.value.shape).toEqual(compactShape)
          expect(firstForSecond.value.state.session.title).not.toBe("stale")
        }
        yield* Fiber.interrupt(first)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("preserves post-fence events queued before the materializing stream is returned", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const updates = yield* PubSub.unbounded<{ shape: DisplayViewShape; state: DisplayState }>()
      yield* Ref.set(setup.entry, yield* setup.makeEntry(
        "generation-1",
        Stream.fromPubSub(updates),
      ))

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const received = yield* Queue.unbounded<StreamEvent>()
        const first = yield* streams.getDisplayViewStream("s1", "timeline", rootShape, true).pipe(
          Stream.tap((event) => Queue.offer(received, event)),
          Stream.runDrain,
          Effect.fork,
        )
        yield* Queue.take(received)
        let warmed = false
        for (let attempt = 0; attempt < 20 && !warmed; attempt += 1) {
          yield* PubSub.publish(updates, { shape: rootShape, state: displayState("warmup") })
          warmed = Option.isSome(yield* Queue.take(received).pipe(Effect.timeoutOption("10 millis")))
        }
        expect(warmed).toBe(true)

        const current = yield* Ref.get(setup.entry)
        if (!current) return yield* Effect.die("missing fake resident")
        const postFence = { shape: compactShape, state: displayState("post-fence") }
        yield* Ref.set(setup.entry, {
          ...current,
          session: {
            ...current.session,
            displayView: {
              ...current.session.displayView,
              setShapeAndSnapshot: (_viewId, shape) => PubSub.publish(updates, postFence).pipe(
                Effect.as({ shape, state: displayState("committed") }),
              ),
            },
          },
        })

        const events = yield* streams.getDisplayViewStream(
          "s1",
          "timeline",
          compactShape,
          true,
        ).pipe(
          Stream.take(2),
          Stream.runCollect,
          Effect.timeoutOption("1 second"),
        )
        expect(Option.isSome(events)).toBe(true)
        if (Option.isNone(events)) return yield* Effect.die("post-fence event was lost")

        const receivedEvents = Array.from(events.value)
        expect(receivedEvents[0]).toEqual({
          _tag: "state",
          shape: compactShape,
          state: displayState("committed"),
        })
        expect(receivedEvents[1]).toEqual({ _tag: "state", ...postFence })
        yield* Fiber.interrupt(first)
      }).pipe(Effect.provide(setup.layer))
    })

    await Effect.runPromise(program)
  })

  it("serializes failed shape admissions against the last committed shape", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const firstReady = yield* Deferred.make<void>()
        const subscriber = yield* streams.getDisplayViewStream("s1", "view-a", rootShape, true).pipe(
          Stream.tap(() => Deferred.succeed(firstReady, undefined)),
          Stream.runDrain,
          Effect.fork,
        )
        yield* Deferred.await(firstReady)

        const healthy = yield* Ref.get(setup.entry)
        if (!healthy) return yield* Effect.die("missing fake resident")
        const failMaterialization = () => Effect.die("shape materialization failed")
        yield* Ref.set(setup.entry, {
          ...healthy,
          session: {
            ...healthy.session,
            displayView: {
              ...healthy.session.displayView,
              setShape: failMaterialization,
              setShapeAndSnapshot: failMaterialization,
            },
          },
        })

        const firstEntered = yield* Deferred.make<void>()
        const firstRelease = yield* Deferred.make<void>()
        yield* Ref.set(setup.withSessionGate, { entered: firstEntered, release: firstRelease })
        const first = yield* streams.setDisplayViewShape("s1", "view-a", compactShape).pipe(
          Effect.exit,
          Effect.fork,
        )
        yield* Deferred.await(firstEntered)

        const secondEntered = yield* Deferred.make<void>()
        const secondRelease = yield* Deferred.make<void>()
        yield* Ref.set(setup.withSessionGate, { entered: secondEntered, release: secondRelease })
        const second = yield* streams.setDisplayViewShape("s1", "view-a", tinyShape).pipe(
          Effect.exit,
          Effect.fork,
        )
        const overlapped = yield* Deferred.await(secondEntered).pipe(Effect.timeoutOption("20 millis"))

        yield* Deferred.succeed(firstRelease, undefined)
        const firstExit = yield* Fiber.join(first)
        expect(Exit.isFailure(firstExit)).toBe(true)
        if (Option.isNone(overlapped)) yield* Deferred.await(secondEntered)
        yield* Deferred.succeed(secondRelease, undefined)
        const secondExit = yield* Fiber.join(second)
        expect(Exit.isFailure(secondExit)).toBe(true)

        yield* Ref.set(setup.withSessionGate, null)
        yield* Ref.set(setup.entry, healthy)
        yield* streams.requestDisplayViewSnapshot("s1", "view-a")
        expect((yield* Ref.get(setup.shapes)).at(-1)).toEqual(rootShape)
        yield* Fiber.interrupt(subscriber)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("keeps ACN attachment state aligned when cancellation arrives during runtime commit", async () => {
    const setup = await Effect.runPromise(makeSetup)
    const program = Effect.gen(function* () {
      const entered = yield* Deferred.make<void>()
      const release = yield* Deferred.make<void>()
      const entry = yield* Ref.get(setup.entry)
      if (!entry) return yield* Effect.die("missing runtime entry")
      yield* Ref.set(setup.entry, {
        ...entry,
        session: {
          ...entry.session,
          displayView: {
            ...entry.session.displayView,
            setShapeAndSnapshot: (_viewId, shape) => Deferred.succeed(entered, undefined).pipe(
              Effect.zipRight(Deferred.await(release)),
              Effect.zipRight(Ref.update(setup.shapes, (all) => [...all, shape])),
              Effect.as({ shape, state: displayState("committed") }),
            ),
          },
        },
      })

      const streams = yield* DisplayViewStreams
      const materializeFiber = yield* streams.setDisplayViewShape(
        "session-1",
        "view-1",
        compactShape,
      ).pipe(Effect.fork)
      yield* Deferred.await(entered)
      const interruptFiber = yield* Fiber.interrupt(materializeFiber).pipe(Effect.fork)
      yield* Deferred.succeed(release, undefined)
      yield* Fiber.join(interruptFiber)

      const snapshot = yield* streams.requestDisplayViewSnapshot("session-1", "view-1")
      expect(snapshot.shape).toEqual(compactShape)
    })
    await Effect.runPromise(program.pipe(Effect.provide(setup.layer)))
  })

  it("uses the runtime's atomic shape-and-snapshot command", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const current = yield* Ref.get(setup.entry)
      if (!current) return yield* Effect.die("missing fake resident")
      const atomicCalls = yield* Ref.make(0)
      yield* Ref.set(setup.entry, {
        ...current,
        session: {
          ...current.session,
          displayView: {
            ...current.session.displayView,
            setShape: () => Effect.die("split setShape must not run"),
            snapshot: () => Effect.die("split snapshot must not run"),
            setShapeAndSnapshot: (_viewId, shape) => Ref.update(
              atomicCalls,
              (count) => count + 1,
            ).pipe(Effect.as({ shape, state: displayState("atomic") })),
          },
        },
      })

      const event = yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        return yield* streams.getDisplayViewStream("s1", "atomic", compactShape, true).pipe(
          Stream.runHead,
        )
      }).pipe(Effect.provide(setup.layer))

      expect(Option.isSome(event)).toBe(true)
      if (Option.isSome(event) && event.value._tag === "state") {
        expect(event.value.shape).toEqual(compactShape)
        expect(event.value.state.session.title).toBe("atomic")
      }
      expect(yield* Ref.get(atomicCalls)).toBe(1)
    })
    await Effect.runPromise(program)
  })

  it("broadcasts every authoritative update to overlapping subscribers", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const upstream = yield* PubSub.unbounded<{ shape: DisplayViewShape; state: DisplayState }>()
      yield* Ref.set(setup.entry, yield* setup.makeEntry("generation-1", Stream.fromPubSub(upstream)))

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const firstReady = yield* Deferred.make<void>()
        const secondReady = yield* Deferred.make<void>()
        const firstUpdates = yield* Queue.unbounded<string | null>()
        const secondUpdates = yield* Queue.unbounded<string | null>()
        const first = yield* streams.getDisplayViewStream("s1", "shared", rootShape, true).pipe(
          Stream.tap((event) => event._tag === "state"
            ? Queue.offer(firstUpdates, event.state.session.title).pipe(
                Effect.zipRight(Deferred.succeed(firstReady, undefined)),
              )
            : Effect.void),
          Stream.runDrain,
          Effect.fork,
        )
        yield* Deferred.await(firstReady)
        const second = yield* streams.getDisplayViewStream("s1", "shared", rootShape).pipe(
          Stream.tap((event) => event._tag === "state"
            ? Queue.offer(secondUpdates, event.state.session.title).pipe(
                Effect.zipRight(Deferred.succeed(secondReady, undefined)),
              )
            : Effect.void),
          Stream.runDrain,
          Effect.fork,
        )
        yield* Deferred.await(secondReady)
        yield* Queue.take(firstUpdates)
        yield* Queue.take(secondUpdates)

        yield* PubSub.publish(upstream, { shape: rootShape, state: displayState("broadcast") })
        expect(yield* Queue.take(firstUpdates)).toBe("broadcast")
        expect(yield* Queue.take(secondUpdates)).toBe("broadcast")
        yield* Fiber.interrupt(second)
        yield* Fiber.interrupt(first)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("refreshes a same-generation materializing readmission", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Ref.set(setup.entry, yield* setup.makeEntry("generation-1", Stream.never))

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const firstReady = yield* Deferred.make<void>()
        const first = yield* streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
          Stream.tap(() => Deferred.succeed(firstReady, undefined)),
          Stream.runDrain,
          Effect.fork,
        )
        yield* Deferred.await(firstReady)

        const second = yield* Effect.raceFirst(
          streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
            Stream.runHead,
            Effect.map((event) => ({ _tag: "event" as const, event })),
          ),
          Effect.sleep("100 millis").pipe(Effect.as({ _tag: "timeout" as const })),
        )
        expect(second._tag).toBe("event")
        if (second._tag === "event") expect(Option.isSome(second.event)).toBe(true)
        yield* Fiber.interrupt(first)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("reattaches when the active stream ends during a same-generation refresh", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const endFirstStream = yield* Deferred.make<void>()
      const firstAttachment = yield* Deferred.make<Fiber.Fiber<{
        readonly shape: DisplayViewShape
        readonly state: DisplayState
      }, never>>()
      const refreshEntered = yield* Deferred.make<void>()
      const releaseRefresh = yield* Deferred.make<void>()
      const liveUpdates = yield* PubSub.unbounded<{ shape: DisplayViewShape; state: DisplayState }>()
      const streamCalls = yield* Ref.make(0)
      const shapeCalls = yield* Ref.make(0)
      const entry = yield* setup.makeEntry("generation-1")
      const firstDisplay = Stream.fromEffect(
        Effect.withFiberRuntime<{
          readonly shape: DisplayViewShape
          readonly state: DisplayState
        }>((fiber) =>
          Deferred.succeed(firstAttachment, fiber).pipe(
            Effect.as({
              shape: rootShape,
              state: displayState("generation-1"),
            }),
          )
        ),
      ).pipe(
        Stream.concat(Stream.fromEffect(Deferred.await(endFirstStream)).pipe(Stream.drain)),
      )
      yield* Ref.set(setup.entry, {
        ...entry,
        session: {
          ...entry.session,
          on: { ...entry.session.on, restoreQueuedMessages: Stream.empty },
          displayView: {
            ...entry.session.displayView,
            stream: () => Stream.unwrap(
              Ref.getAndUpdate(streamCalls, (count) => count + 1).pipe(
                Effect.map((index) => index === 0 ? firstDisplay : Stream.fromPubSub(liveUpdates)),
              ),
            ),
            setShapeAndSnapshot: (_viewId, shape) =>
              Ref.getAndUpdate(shapeCalls, (count) => count + 1).pipe(
                Effect.flatMap((index) => index === 1
                  ? Deferred.succeed(refreshEntered, undefined).pipe(
                      Effect.zipRight(Deferred.await(releaseRefresh)),
                    )
                  : Effect.void),
                Effect.zipRight(Ref.update(setup.shapes, (all) => [...all, shape])),
                Effect.as({ shape, state: displayState("generation-1") }),
              ),
          },
        },
      })

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const received = yield* Queue.unbounded<string>()
        const first = yield* streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
          Stream.runForEach((event) => event._tag === "state"
            ? Queue.offer(received, event.state.session.title ?? "")
            : Effect.void),
          Effect.fork,
        )
        const initial = yield* Effect.raceFirst(
          Queue.take(received).pipe(
            Effect.map((title) => ({ _tag: "event" as const, title })),
          ),
          Effect.sleep("1 second").pipe(Effect.as({ _tag: "timeout" as const })),
        )
        expect(initial).toEqual({ _tag: "event", title: "generation-1" })

        const refresh = yield* streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
          Stream.runHead,
          Effect.fork,
        )
        const refreshStarted = yield* Effect.raceFirst(
          Deferred.await(refreshEntered).pipe(Effect.as(true)),
          Effect.sleep("1 second").pipe(Effect.as(false)),
        )
        expect(refreshStarted).toBe(true)
        const attachment = yield* Deferred.await(firstAttachment)
        yield* Deferred.succeed(endFirstStream, undefined)
        const streamEnded = yield* Effect.raceFirst(
          Fiber.await(attachment).pipe(Effect.as(true)),
          Effect.sleep("1 second").pipe(Effect.as(false)),
        )
        expect(streamEnded).toBe(true)
        // The merge producer has finished. Yield to its already-runnable parent so
        // the attachment finalizer crosses the synchronous ended marker.
        yield* Effect.yieldNow()
        yield* Deferred.succeed(releaseRefresh, undefined)
        const refreshed = yield* Effect.raceFirst(
          Fiber.join(refresh).pipe(
            Effect.map((event) => ({ _tag: "event" as const, event })),
          ),
          Effect.sleep("1 second").pipe(Effect.as({ _tag: "timeout" as const })),
        )
        expect(refreshed._tag).toBe("event")
        if (refreshed._tag === "event") expect(Option.isSome(refreshed.event)).toBe(true)

        expect(yield* Ref.get(streamCalls)).toBe(2)
        yield* Queue.takeAll(received)
        yield* PubSub.publish(liveUpdates, {
          shape: rootShape,
          state: displayState("after-reattach"),
        })
        const update = yield* Effect.raceFirst(
          Queue.take(received).pipe(Effect.map((title) => ({ _tag: "event" as const, title }))),
          Effect.sleep("100 millis").pipe(Effect.as({ _tag: "timeout" as const })),
        )
        expect(update).toEqual({ _tag: "event", title: "after-reattach" })
        yield* Fiber.interrupt(first)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("applies the latest materializing stream shape on readmission", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Ref.set(setup.entry, yield* setup.makeEntry("generation-1", Stream.never))

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const firstReady = yield* Deferred.make<void>()
        const first = yield* streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
          Stream.tap(() => Deferred.succeed(firstReady, undefined)),
          Stream.runDrain,
          Effect.fork,
        )
        yield* Deferred.await(firstReady)

        yield* streams.getDisplayViewStream("s1", "headless", compactShape, true).pipe(
          Stream.runHead,
        )
        expect((yield* Ref.get(setup.shapes)).at(-1)).toEqual(compactShape)
        yield* Fiber.interrupt(first)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("fails a materializing subscriber when the runtime display stream fails", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const failure = new DisplayViewRuntimeError({
        viewId: "failed-stream",
        operation: "stream",
        cause: new Addressed.AddressedCollectionError({
          collection: "display-view",
          operation: "stream",
          reason: "planned failure",
        }),
      })
      const failingStream = Stream.succeed({
        shape: rootShape,
        state: displayState("before-failure"),
      }).pipe(Stream.concat(Stream.fail(failure)))
      yield* Ref.set(setup.entry, yield* setup.makeEntry("failure", failingStream))

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const exit = yield* streams
          .getDisplayViewStream("s1", "failed-stream", rootShape, true)
          .pipe(Stream.runDrain, Effect.exit)
        expect(Exit.isFailure(exit)).toBe(true)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("rolls back a failed materializing stream shape intent", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const firstReady = yield* Deferred.make<void>()
        const first = yield* streams.getDisplayViewStream("s1", "headless", rootShape, true).pipe(
          Stream.tap(() => Deferred.succeed(firstReady, undefined)),
          Stream.runDrain,
          Effect.fork,
        )
        yield* Deferred.await(firstReady)

        const healthy = yield* Ref.get(setup.entry)
        if (!healthy) return yield* Effect.die("missing fake resident")
        yield* Ref.set(setup.entry, {
          ...healthy,
          session: {
            ...healthy.session,
            displayView: {
              ...healthy.session.displayView,
              setShape: () => Effect.die("shape readmission failed"),
              setShapeAndSnapshot: () => Effect.die("shape readmission failed"),
            },
          },
        })
        const failed = yield* streams
          .getDisplayViewStream("s1", "headless", compactShape, true)
          .pipe(Stream.runHead, Effect.exit)
        expect(Exit.isFailure(failed)).toBe(true)

        yield* Ref.set(setup.entry, healthy)
        yield* streams.requestDisplayViewSnapshot("s1", "headless")
        expect((yield* Ref.get(setup.shapes)).at(-1)).toEqual(rootShape)
        yield* Fiber.interrupt(first)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("removes a registration when subscriber admission fails", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const failing = yield* setup.makeEntry("generation-1")
      yield* Ref.set(setup.entry, {
        ...failing,
        session: {
          ...failing.session,
          displayView: {
            ...failing.session.displayView,
            setShape: () => Effect.die("shape admission failed"),
            setShapeAndSnapshot: () => Effect.die("shape admission failed"),
          },
        },
      })
      yield* Ref.set(setup.busy, true)

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const admission = yield* streams.getDisplayViewStream("s1", "failed", rootShape, true).pipe(
          Stream.runDrain,
          Effect.exit,
        )
        expect(admission._tag).toBe("Failure")

        yield* Ref.set(setup.entry, yield* setup.makeEntry("generation-1"))
        const resync = yield* streams.requestDisplayViewSnapshot("s1", "failed").pipe(Effect.exit)
        expect(resync._tag).toBe("Failure")
        if (resync._tag === "Failure" && resync.cause._tag === "Fail") {
          expect(resync.cause.error._tag).toBe("DisplayViewNotOpen")
        }
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("removes the final subscriber registration even when runtime display close fails", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const current = yield* Ref.get(setup.entry)
      if (!current) return yield* Effect.die("missing fake resident")
      yield* Ref.set(setup.entry, {
        ...current,
        session: {
          ...current.session,
          displayView: {
            ...current.session.displayView,
            close: () => Effect.die("injected close failure"),
          },
        },
      })

      const snapshotExit = yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const fiber = yield* streams
          .getDisplayViewStream("s1", "headless", rootShape, true)
          .pipe(Stream.runDrain, Effect.fork)
        for (let attempt = 0; attempt < 1_000; attempt++) {
          if ((yield* Ref.get(setup.shapes)).length > 0) break
          yield* Effect.yieldNow()
        }
        expect((yield* Ref.get(setup.shapes)).length).toBeGreaterThan(0)
        yield* Ref.set(setup.busy, true)
        yield* Fiber.interrupt(fiber)
        return yield* streams.requestDisplayViewSnapshot("s1", "headless").pipe(Effect.exit)
      }).pipe(Effect.provide(setup.layer))

      expect(Exit.isFailure(snapshotExit)).toBe(true)
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

  it("does not let a runtime-change scan attach after its final subscriber leaves", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const entered = yield* Deferred.make<void>()
      const release = yield* Deferred.make<void>()
      yield* Ref.set(setup.busyAttachGate, { entered, release })

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const subscriber = yield* streams
          .getDisplayViewStream("s1", "view-a", rootShape)
          .pipe(Stream.runDrain, Effect.fork)
        for (let attempt = 0; attempt < 10; attempt++) yield* Effect.yieldNow()

        yield* Ref.set(setup.busy, true)
        yield* PubSub.publish(setup.changes, undefined)
        yield* Deferred.await(entered)

        const interrupt = yield* Fiber.interrupt(subscriber).pipe(Effect.fork)
        const cleanupCompleted = yield* Effect.raceFirst(
          Fiber.join(interrupt).pipe(Effect.as(true)),
          Effect.sleep("100 millis").pipe(Effect.as(false)),
        )

        yield* Deferred.succeed(release, undefined)
        yield* Fiber.join(interrupt)
        expect(cleanupCompleted).toBe(true)
        expect(yield* Ref.get(setup.shapes)).toHaveLength(0)
        expect(yield* Ref.get(setup.closed)).toEqual(["view-a"])
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("does not hold registration serialization while resident close admission waits", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const entered = yield* Deferred.make<void>()
      const release = yield* Deferred.make<void>()
      yield* Ref.set(setup.busyCloseGate, { entered, release })
      const scope = yield* Scope.make()
      const context = yield* Layer.buildWithScope(setup.layer, scope)
      const streams = Context.get(context, DisplayViewStreams)
      const ready = yield* Deferred.make<void>()
      const subscriber = yield* streams
        .getDisplayViewStream("s1", "view-a", rootShape, true)
        .pipe(
          Stream.tap(() => Deferred.succeed(ready, undefined)),
          Stream.runDrain,
          Effect.fork,
        )
      yield* Deferred.await(ready)
      yield* Ref.set(setup.busy, true)

      const interrupt = yield* Fiber.interrupt(subscriber).pipe(Effect.fork)
      yield* Deferred.await(entered)
      const shutdown = yield* Scope.close(scope, Exit.void).pipe(Effect.fork)
      const shutdownCompleted = yield* Effect.raceFirst(
        Fiber.join(shutdown).pipe(Effect.as(true)),
        Effect.sleep("100 millis").pipe(Effect.as(false)),
      )

      yield* Deferred.succeed(release, undefined)
      yield* Fiber.join(interrupt)
      yield* Fiber.join(shutdown)
      expect(shutdownCompleted).toBe(true)
    })
    await Effect.runPromise(program)
  })

  it("does not hold display serialization while runtime acquisition is pending", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const firstEvent = yield* Deferred.make<void>()
        const subscriber = yield* streams
          .getDisplayViewStream("s1", "view-a", rootShape, true)
          .pipe(
            Stream.tap(() => Deferred.succeed(firstEvent, undefined)),
            Stream.runDrain,
            Effect.fork,
          )
        yield* Deferred.await(firstEvent)

        const entered = yield* Deferred.make<void>()
        const release = yield* Deferred.make<void>()
        yield* Ref.set(setup.withSessionGate, { entered, release })
        const resync = yield* streams.requestDisplayViewSnapshot("s1", "view-a").pipe(Effect.fork)
        yield* Deferred.await(entered)

        const observers = yield* Ref.get(setup.observers)
        expect(observers.size).toBe(1)
        const retirementCompleted = yield* Effect.raceFirst(
          Effect.forEach(
            observers,
            (observer) => observer.retire({ sessionId: "s1", generation: 1 }),
            { discard: true },
          ).pipe(Effect.as(true)),
          Effect.sleep("100 millis").pipe(Effect.as(false)),
        )

        yield* Ref.set(setup.generation, 2)
        yield* Deferred.succeed(release, undefined)
        yield* Fiber.join(resync)
        yield* Fiber.interrupt(subscriber)
        expect(retirementCompleted).toBe(true)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("serializes shape intent across delayed runtime acquisition", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const firstEvent = yield* Deferred.make<void>()
        const subscriber = yield* streams
          .getDisplayViewStream("s1", "view-a", rootShape, true)
          .pipe(
            Stream.tap(() => Deferred.succeed(firstEvent, undefined)),
            Stream.runDrain,
            Effect.fork,
          )
        yield* Deferred.await(firstEvent)

        const entered = yield* Deferred.make<void>()
        const release = yield* Deferred.make<void>()
        yield* Ref.set(setup.withSessionGate, { entered, release })
        const older = yield* streams.setDisplayViewShape("s1", "view-a", compactShape).pipe(Effect.fork)
        yield* Deferred.await(entered)

        yield* Ref.set(setup.withSessionGate, null)
        const newer = yield* streams.setDisplayViewShape("s1", "view-a", rootShape).pipe(Effect.fork)
        yield* Effect.yieldNow()
        expect(Option.isNone(yield* Fiber.poll(newer))).toBe(true)
        yield* Deferred.succeed(release, undefined)
        yield* Fiber.join(older)
        yield* Fiber.join(newer)

        yield* streams.requestDisplayViewSnapshot("s1", "view-a")
        expect((yield* Ref.get(setup.shapes)).at(-1)).toEqual(rootShape)
        yield* Fiber.interrupt(subscriber)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("does not close a replacement registration while removing its predecessor", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup

      yield* Effect.gen(function* () {
        const streams = yield* DisplayViewStreams
        const firstEvent = yield* Deferred.make<void>()
        const first = yield* streams
          .getDisplayViewStream("s1", "view-a", rootShape, true)
          .pipe(
            Stream.tap(() => Deferred.succeed(firstEvent, undefined)),
            Stream.runDrain,
            Effect.fork,
          )
        yield* Deferred.await(firstEvent)
        yield* Ref.set(setup.busy, true)

        const entered = yield* Deferred.make<void>()
        const release = yield* Deferred.make<void>()
        yield* Ref.set(setup.closeGate, { entered, release })
        const stopping = yield* Fiber.interrupt(first).pipe(Effect.fork)
        const closeStarted = yield* Effect.raceFirst(
          Deferred.await(entered).pipe(Effect.as(true)),
          Effect.sleep("500 millis").pipe(Effect.as(false)),
        )
        expect(closeStarted).toBe(true)

        const replacement = yield* streams
          .getDisplayViewStream("s1", "view-a", compactShape, true)
          .pipe(Stream.runHead, Effect.fork)
        for (let attempt = 0; attempt < 100; attempt++) yield* Effect.yieldNow()
        const replacementWasPending = Option.isNone(yield* Fiber.poll(replacement))

        yield* Deferred.succeed(release, undefined)
        yield* Fiber.join(stopping)
        const replacementResult = yield* Fiber.join(replacement)
        expect(replacementWasPending).toBe(true)
        expect(Option.isSome(replacementResult)).toBe(true)
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
        yield* streams.setDisplayViewShape("s1", "view-a", rootShape)
        const observer = [...(yield* Ref.get(setup.observers))][0]
        if (!observer) return yield* Effect.die("missing retirement observer")

        const retired = yield* Effect.raceFirst(
          observer.retire({ sessionId: "s1", generation: 1 }).pipe(Effect.as(true)),
          Effect.sleep("100 millis").pipe(Effect.as(false)),
        )
        yield* Deferred.succeed(releaseFinalizer, undefined)
        expect(retired).toBe(true)
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
