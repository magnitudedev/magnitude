import { describe, expect, it } from "vitest"
import {
  Deferred,
  Effect,
  Either,
  Fiber,
  Layer,
  Option,
  PubSub,
  Ref,
  Scope,
  Stream,
  TestClock,
  TestContext,
} from "effect"
import type {
  AgentLifecycleState,
  CodingAgentSession,
  ForkTurnState,
  SessionWorkStatus,
} from "@magnitudedev/agent"
import type { StoredSessionMeta } from "@magnitudedev/storage"
import { SessionOperationFailed } from "@magnitudedev/protocol"
import { AgentFactory, type AgentFactoryApi } from "./agent-factory"
import {
  AgentRuntime,
  makeAgentRuntimeLive,
  type AgentRuntimeApi,
  type RuntimeStartRequest,
} from "./agent-runtime"
import { SessionStore, type SessionStoreApi } from "./session-store"
import {
  normalizeSessionRuntimeOptions,
  SessionRuntimeOptionsStore,
  type SessionRuntimeOptions,
  type SessionRuntimeOptionsStoreApi,
} from "./session-runtime-options"

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

const idleSession: CodingAgentSession = {
  on: { restoreQueuedMessages: Stream.never },
  state: {
    work: {
      get: () => Effect.succeed({ _tag: "Quiescent" as const, workerCount: 0 }),
      subscribe: Stream.succeed({ _tag: "Quiescent" as const, workerCount: 0 }),
    },
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
    stream: () => Stream.never,
    snapshot: () => Effect.die("unused"),
    setShape: () => Effect.die("unused"),
    close: () => Effect.void,
  },
  send: () => Effect.die("unused"),
  interrupt: () => Effect.die("unused"),
  publishInitialTask: () => Effect.void,
  onEvent: Stream.never,
  onError: Stream.never,
  subscribeIntrospection: () => Stream.never,
}

const makeMeta = (sessionId: string, cwd = "/repo"): StoredSessionMeta => {
  const now = new Date().toISOString()
  return {
    sessionId,
    created: now,
    updated: now,
    chatName: "Session",
    workingDirectory: cwd,
    visibility: "visible",
    initialVersion: "0.0.1",
    lastActiveVersion: "0.0.1",
    gitBranch: null,
    firstUserMessage: null,
    lastMessage: null,
    messageCount: 0,
  }
}

const request = (sessionId: string): RuntimeStartRequest => ({
  sessionId,
  cwd: "/repo",
  options: normalizeSessionRuntimeOptions(),
  visibility: "visible",
})

const residentCount = (runtime: AgentRuntimeApi) =>
  runtime.residentSessions.pipe(Effect.map((sessions) => sessions.length))

const makeLayer = (input: {
  readonly factory: AgentFactoryApi
  readonly storedSessions?: ReadonlyArray<StoredSessionMeta>
  readonly storedRuntimeOptions?: ReadonlyMap<string, SessionRuntimeOptions>
}) => {
  const dependencies = Layer.mergeAll(
    Layer.succeed(AgentFactory, input.factory),
    Layer.effect(
      SessionStore,
      Effect.gen(function* () {
        const metas = yield* Ref.make(
          new Map((input.storedSessions ?? []).map((meta) => [meta.sessionId, meta])),
        )
        return {
          createId: Effect.die("unused"),
          readMeta: (sessionId) =>
            Ref.get(metas).pipe(Effect.map((all) => all.get(sessionId) ?? null)),
          readProtocolMeta: () => Effect.die("unused"),
          promoteDraft: () => Effect.die("unused"),
          listDraftSessionIds: () => Effect.die("unused"),
          listProtocolMetas: () => Effect.die("unused"),
          listSessionCwds: () => Effect.die("unused"),
          deleteSessionFiles: () => Effect.die("unused"),
          validateCwd: Effect.succeed,
          getScratchpadPath: (sessionId) => Effect.succeed(`/tmp/${sessionId}/scratchpad`),
          getExecutionContext: () => Effect.die("unused"),
        } satisfies SessionStoreApi
      }),
    ),
    Layer.effect(
      SessionRuntimeOptionsStore,
      Effect.gen(function* () {
        const values = yield* Ref.make(new Map(input.storedRuntimeOptions ?? []))
        return {
          normalize: normalizeSessionRuntimeOptions,
          read: (sessionId) =>
            Ref.get(values).pipe(Effect.map((all) => all.get(sessionId) ?? null)),
          write: (sessionId, options) =>
            Ref.update(values, (all) => new Map(all).set(sessionId, options)),
        } satisfies SessionRuntimeOptionsStoreApi
      }),
    ),
  )
  return makeAgentRuntimeLive({ idleTimeout: "2 seconds" }).pipe(
    Layer.provide(dependencies),
    Layer.provideMerge(TestContext.TestContext),
  )
}

describe("AgentRuntime", () => {
  it("single-flights startup and publishes one generation", async () => {
    const program = Effect.gen(function* () {
      const calls = yield* Ref.make(0)
      const entered = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      const layer = makeLayer({
        factory: {
          createSession: () =>
            Ref.update(calls, (value) => value + 1).pipe(
              Effect.zipRight(Deferred.succeed(entered, undefined)),
              Effect.zipRight(Deferred.await(resume)),
              Effect.as(idleSession),
            ),
        },
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const first = yield* runtime
          .withSessionRequest(request("single"), "first", (_, generation) =>
            Effect.succeed(generation),
          )
          .pipe(Effect.fork)
        yield* Deferred.await(entered)
        const second = yield* runtime
          .withSessionRequest(request("single"), "second", (_, generation) =>
            Effect.succeed(generation),
          )
          .pipe(Effect.fork)
        yield* Deferred.succeed(resume, undefined)
        expect(yield* Fiber.join(first)).toBe(1)
        expect(yield* Fiber.join(second)).toBe(1)
        expect(yield* Ref.get(calls)).toBe(1)
        expect(yield* residentCount(runtime)).toBe(1)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })

  it("holds in-flight work, then evicts at the exact post-release deadline", async () => {
    const program = Effect.gen(function* () {
      const latch = yield* Deferred.make<void>()
      const layer = makeLayer({ factory: { createSession: () => Effect.succeed(idleSession) } })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const inFlight = yield* runtime
          .withSessionRequest(request("deadline"), "blocked", () => Deferred.await(latch))
          .pipe(Effect.fork)
        yield* Effect.yieldNow()
        yield* TestClock.adjust("1 hour")
        expect(yield* residentCount(runtime)).toBe(1)
        yield* Deferred.succeed(latch, undefined)
        yield* Fiber.join(inFlight)
        yield* TestClock.adjust("1999 millis")
        expect(yield* residentCount(runtime)).toBe(1)
        yield* TestClock.adjust("1 millis")
        yield* Effect.yieldNow()
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })

  it("holds continuing session work and starts a fresh idle interval at quiescence", async () => {
    const program = Effect.gen(function* () {
      const status = yield* Ref.make<SessionWorkStatus>({
        _tag: "Quiescent",
        workerCount: 0,
      })
      const changes = yield* PubSub.unbounded<SessionWorkStatus>()
      const working: SessionWorkStatus = { _tag: "Working", workerCount: 1 }
      const quiescent: SessionWorkStatus = { _tag: "Quiescent", workerCount: 0 }
      const session: CodingAgentSession = {
        ...idleSession,
        state: {
          ...idleSession.state,
          work: {
            get: () => Ref.get(status),
            subscribe: Stream.fromPubSub(changes),
          },
        },
      }
      const layer = makeLayer({ factory: { createSession: () => Effect.succeed(session) } })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* runtime.withSessionRequest(request("continuing"), "start-work", () =>
          Ref.set(status, working).pipe(
            Effect.zipRight(PubSub.publish(changes, working)),
          ),
        )
        yield* Effect.yieldNow()
        yield* TestClock.adjust("1 hour")
        expect(yield* residentCount(runtime)).toBe(1)

        yield* Ref.set(status, quiescent)
        yield* PubSub.publish(changes, quiescent)
        yield* Effect.yieldNow()
        yield* TestClock.adjust("1999 millis")
        expect(yield* residentCount(runtime)).toBe(1)
        yield* TestClock.adjust("1 millis")
        yield* Effect.yieldNow()
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })

  it("rehydrates with a new generation after eviction", async () => {
    const program = Effect.gen(function* () {
      const layer = makeLayer({
        factory: { createSession: () => Effect.succeed(idleSession) },
        storedSessions: [makeMeta("rehydrate")],
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        expect(yield* runtime.withSession("rehydrate", "one", (_, generation) => Effect.succeed(generation))).toBe(1)
        yield* TestClock.adjust("2 seconds")
        yield* Effect.yieldNow()
        expect(yield* residentCount(runtime)).toBe(0)
        expect(yield* runtime.withSession("rehydrate", "two", (_, generation) => Effect.succeed(generation))).toBe(2)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })

  it("does not let passive busy-joins revive or prolong an idle generation", async () => {
    const program = Effect.gen(function* () {
      const layer = makeLayer({ factory: { createSession: () => Effect.succeed(idleSession) } })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* runtime.withSessionRequest(request("passive"), "demand", () => Effect.void)
        yield* TestClock.adjust("1 second")
        const observed = yield* runtime.tryWithBusyResident(
          "passive",
          "ambient",
          () => Effect.succeed("unexpected"),
        )
        expect(Option.isNone(observed)).toBe(true)
        yield* TestClock.adjust("1 second")
        yield* Effect.yieldNow()
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })

  it("clears a failed startup so a later request can retry", async () => {
    const program = Effect.gen(function* () {
      const calls = yield* Ref.make(0)
      const failure = new SessionOperationFailed({ operation: "start", reason: "boom" })
      const layer = makeLayer({
        factory: {
          createSession: () =>
            Ref.updateAndGet(calls, (value) => value + 1).pipe(
              Effect.flatMap((call) => (call === 1 ? Effect.fail(failure) : Effect.succeed(idleSession))),
            ),
        },
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const failed = yield* Effect.either(
          runtime.withSessionRequest(request("retry"), "first", () => Effect.void),
        )
        expect(Either.isLeft(failed)).toBe(true)
        expect(yield* runtime.withSessionRequest(request("retry"), "second", (_, generation) => Effect.succeed(generation))).toBe(2)
        expect(yield* Ref.get(calls)).toBe(2)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })

  it("clears an interrupted startup so a later request can retry", async () => {
    const program = Effect.gen(function* () {
      const calls = yield* Ref.make(0)
      const entered = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      const layer = makeLayer({
        factory: {
          createSession: () =>
            Ref.updateAndGet(calls, (value) => value + 1).pipe(
              Effect.flatMap((call) =>
                call === 1
                  ? Deferred.succeed(entered, undefined).pipe(
                      Effect.zipRight(Deferred.await(resume)),
                      Effect.as(idleSession),
                    )
                  : Effect.succeed(idleSession),
              ),
            ),
        },
      })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const first = yield* runtime
          .withSessionRequest(request("interrupted"), "first", () => Effect.void)
          .pipe(Effect.fork)
        yield* Deferred.await(entered)
        yield* Fiber.interrupt(first)
        expect(
          yield* runtime.withSessionRequest(
            request("interrupted"),
            "second",
            (_, generation) => Effect.succeed(generation),
          ),
        ).toBe(2)
        expect(yield* Ref.get(calls)).toBe(2)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })

  it("runs generation finalizers before publishing retirement", async () => {
    const program = Effect.gen(function* () {
      const finalized = yield* Ref.make(false)
      const layer = makeLayer({
        factory: {
          createSession: ({ scope }) =>
            Scope.addFinalizer(scope, Ref.set(finalized, true)).pipe(Effect.as(idleSession)),
        },
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* runtime.withSessionRequest(request("finalize"), "use", () => Effect.void)
        yield* runtime.dispose("finalize")
        expect(yield* Ref.get(finalized)).toBe(true)
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })

  it("closes a working session when the runtime scope closes", async () => {
    const finalized = await Effect.runPromise(
      Effect.gen(function* () {
        const finalized = yield* Ref.make(false)
        const workingSession: CodingAgentSession = {
          ...idleSession,
          state: {
            ...idleSession.state,
            work: {
              get: () => Effect.succeed({ _tag: "Working" as const, workerCount: 1 }),
              subscribe: Stream.never,
            },
          },
        }
        const layer = makeLayer({
          factory: {
            createSession: ({ scope }) =>
              Scope.addFinalizer(scope, Ref.set(finalized, true)).pipe(Effect.as(workingSession)),
          },
        })

        yield* Effect.gen(function* () {
          const runtime = yield* AgentRuntime
          yield* runtime.withSessionRequest(request("working-shutdown"), "start", () => Effect.void)
        }).pipe(Effect.provide(layer))

        return yield* Ref.get(finalized)
      }),
    )

    expect(finalized).toBe(true)
  })

  it("excludes new admission while deletion drains and finalizes the resident generation", async () => {
    const program = Effect.gen(function* () {
      const useEntered = yield* Deferred.make<void>()
      const releaseUse = yield* Deferred.make<void>()
      const removalEntered = yield* Deferred.make<void>()
      const allowRemoval = yield* Deferred.make<void>()
      const finalized = yield* Ref.make(false)
      const finalizedBeforeRemoval = yield* Ref.make(false)
      const layer = makeLayer({
        factory: {
          createSession: ({ scope }) =>
            Scope.addFinalizer(scope, Ref.set(finalized, true)).pipe(Effect.as(idleSession)),
        },
        storedSessions: [makeMeta("delete")],
      })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const inFlight = yield* runtime
          .withSession("delete", "in-flight", () =>
            Deferred.succeed(useEntered, undefined).pipe(
              Effect.zipRight(Deferred.await(releaseUse)),
            ),
          )
          .pipe(Effect.fork)
        yield* Deferred.await(useEntered)
        const deletion = yield* runtime
          .deleteSession(
            "delete",
            Ref.get(finalized).pipe(
              Effect.tap((value) => Ref.set(finalizedBeforeRemoval, value)),
              Effect.zipRight(Deferred.succeed(removalEntered, undefined)),
              Effect.zipRight(Deferred.await(allowRemoval)),
            ),
          )
          .pipe(Effect.fork)
        yield* Effect.yieldNow()

        const rejected = yield* Effect.either(
          runtime.withSession("delete", "too-late", () => Effect.void),
        )
        expect(Either.isLeft(rejected)).toBe(true)
        if (Either.isLeft(rejected)) expect(rejected.left._tag).toBe("SessionOperationFailed")

        yield* Deferred.succeed(releaseUse, undefined)
        yield* Fiber.join(inFlight)
        yield* Deferred.await(removalEntered)
        expect(yield* Ref.get(finalizedBeforeRemoval)).toBe(true)
        expect(yield* residentCount(runtime)).toBe(0)
        yield* Deferred.succeed(allowRemoval, undefined)
        yield* Fiber.join(deletion)
      }).pipe(Effect.provide(layer))
    })
    await Effect.runPromise(program)
  })
})
