import { mkdtemp } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, it } from "vitest"
import {
  Context,
  Deferred,
  Duration,
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
import { MagnitudeStorage, type StoredSessionMeta } from "@magnitudedev/storage"
import {
  DirectoryPathSchema,
  SessionOperationFailed,
  type DirectoryPath,
} from "@magnitudedev/acn-protocol"
import { AgentFactory, type AgentFactoryApi } from "./agent-factory"
import {
  AgentRuntime,
  makeAgentRuntimeLive,
  type AgentRuntimeApi,
  type RuntimeStartRequest,
} from "./agent-runtime"
import {
  normalizeSessionRuntimeOptions,
  SessionRuntimeOptionsStore,
  type SessionRuntimeOptions,
  type SessionRuntimeOptionsStoreApi,
} from "./session-runtime-options"
import { makeTestStorageLayer, testFileSystemManagerLayer } from "./session-test-support"

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

const makeMeta = (sessionId: string, cwd: DirectoryPath): StoredSessionMeta => {
  const now = new Date().toISOString()
  return {
    sessionId,
    archived: false,
    pinnedAt: Option.none(),
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

const residentCount = (runtime: AgentRuntimeApi) =>
  runtime.sessionRuntimes.pipe(Effect.map((sessions) => sessions.length))

const awaitRuntimeWorkStatus = (
  runtime: AgentRuntimeApi,
  tag: SessionWorkStatus["_tag"],
): Effect.Effect<void> => Effect.suspend(() =>
  runtime.sessionRuntimes.pipe(
    Effect.flatMap((sessions) =>
      sessions[0]?.workStatus._tag === tag
        ? Effect.void
        : Effect.yieldNow().pipe(
            Effect.zipRight(awaitRuntimeWorkStatus(runtime, tag)),
          ),
    ),
  ),
)

interface TestSetup {
  readonly cwd: DirectoryPath
  readonly layer: Layer.Layer<AgentRuntime>
  readonly request: (sessionId: string) => RuntimeStartRequest
}

const makeSetup = (input: {
  readonly factory: AgentFactoryApi
  readonly storedSessionIds?: ReadonlyArray<string>
  readonly storedRuntimeOptions?: ReadonlyMap<string, SessionRuntimeOptions>
  readonly retirementAdmissionTimeout?: Duration.DurationInput
  readonly retirementShutdownTimeout?: Duration.DurationInput
}): Effect.Effect<TestSetup> =>
  Effect.gen(function* () {
    const root = yield* Effect.promise(() => mkdtemp(join(tmpdir(), "magnitude-runtime-")))
    const cwd = DirectoryPathSchema.make(root)
    const storageLayer = makeTestStorageLayer(root)
    const dependencies = Layer.mergeAll(
      Layer.succeed(AgentFactory, input.factory),
      storageLayer,
      testFileSystemManagerLayer,
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
    const withSeed = Layer.tap(dependencies, (context) => Effect.gen(function* () {
      const storage = Context.get(context, MagnitudeStorage)
      for (const sessionId of input.storedSessionIds ?? []) {
        yield* storage.sessions.writeMeta(sessionId, makeMeta(sessionId, cwd))
      }
    }))
    const layer = makeAgentRuntimeLive({
      idleTimeout: "2 seconds",
      retirementAdmissionTimeout: input.retirementAdmissionTimeout,
      retirementShutdownTimeout: input.retirementShutdownTimeout,
    }).pipe(
      Layer.provide(Layer.orDie(withSeed)),
      Layer.provideMerge(TestContext.TestContext),
    )
    return {
      cwd,
      layer,
      request: (sessionId): RuntimeStartRequest => ({
        sessionId,
        cwd,
        options: normalizeSessionRuntimeOptions(),
        visibility: "visible",
      }),
    }
  })

describe("AgentRuntime", () => {
  it("single-flights startup and publishes one generation", async () => {
    const program = Effect.gen(function* () {
      const calls = yield* Ref.make(0)
      const entered = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      const setup = yield* makeSetup({
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
          .acquireSessionRequest(setup.request("single"), "first")
          .pipe(Effect.map(({ generation }) => generation), Effect.scoped, Effect.fork)
        yield* Deferred.await(entered)
        const second = yield* runtime
          .acquireSessionRequest(setup.request("single"), "second")
          .pipe(Effect.map(({ generation }) => generation), Effect.scoped, Effect.fork)
        yield* Deferred.succeed(resume, undefined)
        expect(yield* Fiber.join(first)).toBe(1)
        expect(yield* Fiber.join(second)).toBe(1)
        expect(yield* Ref.get(calls)).toBe(1)
        expect(yield* residentCount(runtime)).toBe(1)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("holds in-flight work, then evicts at the exact post-release deadline", async () => {
    const program = Effect.gen(function* () {
      const latch = yield* Deferred.make<void>()
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(idleSession) },
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const inFlight = yield* runtime
          .acquireSessionRequest(setup.request("deadline"), "blocked")
          .pipe(Effect.zipRight(Deferred.await(latch)), Effect.scoped, Effect.fork)
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
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("releases scoped runtime use when the caller is interrupted", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(idleSession) },
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const acquired = yield* Deferred.make<void>()
        const caller = yield* Effect.scoped(
          Effect.gen(function* () {
            yield* runtime.acquireSessionRequest(setup.request("interrupted-use"), "caller")
            yield* Deferred.succeed(acquired, undefined)
            return yield* Effect.never
          }),
        ).pipe(Effect.fork)
        yield* Deferred.await(acquired)
        expect((yield* runtime.sessionRuntimes)[0]?.scopedUseCount).toBe(1)
        yield* Fiber.interrupt(caller)
        expect((yield* runtime.sessionRuntimes)[0]?.scopedUseCount).toBe(0)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("uses authoritative work status to retain the runtime and starts a fresh idle interval at quiescence", async () => {
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
            subscribe: Stream.concat(
              Stream.fromEffect(Ref.get(status)),
              Stream.fromPubSub(changes),
            ),
          },
        },
      }
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(session) },
        storedSessionIds: ["continuing"],
      })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* Effect.scoped(runtime.acquireSession("continuing", "initial-use"))
        yield* Ref.set(status, working)
        yield* PubSub.publish(changes, working)
        yield* awaitRuntimeWorkStatus(runtime, "Working")
        yield* TestClock.adjust("1 hour")
        expect(yield* residentCount(runtime)).toBe(1)

        yield* Ref.set(status, quiescent)
        yield* PubSub.publish(changes, quiescent)
        yield* awaitRuntimeWorkStatus(runtime, "Quiescent")
        yield* TestClock.adjust("1999 millis")
        expect(yield* residentCount(runtime)).toBe(1)
        yield* TestClock.adjust("1 millis")
        yield* Effect.yieldNow()
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("rehydrates with a new generation after eviction", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(idleSession) },
        storedSessionIds: ["rehydrate"],
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        expect(yield* runtime.acquireSession("rehydrate", "one").pipe(
          Effect.map(({ generation }) => generation),
          Effect.scoped,
        )).toBe(1)
        yield* TestClock.adjust("2 seconds")
        yield* Effect.yieldNow()
        expect(yield* residentCount(runtime)).toBe(0)
        expect(yield* runtime.acquireSession("rehydrate", "two").pipe(
          Effect.map(({ generation }) => generation),
          Effect.scoped,
        )).toBe(2)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("bounds admission behind a stalled retirement without blocking other sessions", async () => {
    const program = Effect.gen(function* () {
      const retirementStarted = yield* Deferred.make<void>()
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(idleSession) },
        storedSessionIds: ["stalled", "independent"],
        retirementAdmissionTimeout: "1 second",
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* runtime
          .registerRetirementObserver({
            retire: ({ sessionId }) =>
              sessionId === "stalled"
                ? Deferred.succeed(retirementStarted, undefined).pipe(
                    Effect.zipRight(Effect.never),
                  )
                : Effect.void,
          })
          .pipe(Effect.asVoid)
        yield* Effect.scoped(runtime.acquireSession("stalled", "initial"))
        yield* TestClock.adjust("2 seconds")
        yield* Deferred.await(retirementStarted)

        const blocked = yield* runtime
          .acquireSession("stalled", "after-idle")
          .pipe(Effect.scoped, Effect.either, Effect.fork)
        expect(
          yield* runtime.acquireSession("independent", "unrelated").pipe(
            Effect.map(({ generation }) => generation),
            Effect.scoped,
          ),
        ).toBe(1)

        yield* TestClock.adjust("1 second")
        const result = yield* Fiber.join(blocked)
        expect(Either.isLeft(result)).toBe(true)
        if (Either.isLeft(result)) {
          expect(result.left._tag).toBe("SessionOperationFailed")
          if (result.left._tag === "SessionOperationFailed") {
            expect(result.left.reason).toContain("did not finish shutting down")
          }
        }
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("contains stalled retirement to its exact session", async () => {
    const program = Effect.gen(function* () {
      const retirementStarted = yield* Deferred.make<void>()
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(idleSession) },
        storedSessionIds: ["stalled-shutdown"],
        retirementAdmissionTimeout: "1 second",
        retirementShutdownTimeout: "3 seconds",
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* runtime
          .registerRetirementObserver({
            retire: () =>
              Deferred.succeed(retirementStarted, undefined).pipe(
                Effect.zipRight(Effect.never),
              ),
          })
          .pipe(Effect.asVoid)
        yield* Effect.scoped(runtime.acquireSession("stalled-shutdown", "initial"))
        yield* TestClock.adjust("2 seconds")
        yield* Deferred.await(retirementStarted)
        yield* TestClock.adjust("3 seconds")
        yield* Effect.yieldNow()
        expect(
          yield* Effect.scoped(
            runtime.acquireSessionRequest(setup.request("unrelated"), "unrelated"),
          ).pipe(Effect.map((entry) => entry.entry.id)),
        ).toBe("unrelated")

        const blocked = yield* runtime
          .acquireSession("stalled-shutdown", "blocked")
          .pipe(Effect.scoped, Effect.either, Effect.fork)
        yield* TestClock.adjust("1 second")
        const blockedResult = yield* Fiber.join(blocked)
        expect(Either.isLeft(blockedResult)).toBe(true)
        if (Either.isLeft(blockedResult)) {
          expect(blockedResult.left._tag).toBe("SessionOperationFailed")
        }
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("completes successful retirement before its watchdog", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(idleSession) },
        retirementShutdownTimeout: "3 seconds",
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* Effect.scoped(
          runtime.acquireSessionRequest(setup.request("successful-retirement"), "initial"),
        )
        yield* runtime.dispose("successful-retirement")
        yield* TestClock.adjust("3 seconds")
        yield* Effect.yieldNow()
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("does not let passive active-only acquisition revive an idle runtime", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(idleSession) },
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* Effect.scoped(
          runtime.acquireSessionRequest(setup.request("passive"), "initial-use"),
        )
        yield* TestClock.adjust("1 second")
        const observed = yield* Effect.scoped(
          runtime.tryAcquireActiveSession("passive", "ambient"),
        )
        expect(Option.isNone(observed)).toBe(true)
        yield* TestClock.adjust("1 second")
        yield* Effect.yieldNow()
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("clears a failed startup so a later request can retry", async () => {
    const program = Effect.gen(function* () {
      const calls = yield* Ref.make(0)
      const failure = new SessionOperationFailed({ operation: "start", reason: "boom" })
      const setup = yield* makeSetup({
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
          Effect.scoped(runtime.acquireSessionRequest(setup.request("retry"), "first")),
        )
        expect(Either.isLeft(failed)).toBe(true)
        expect(yield* runtime.acquireSessionRequest(setup.request("retry"), "second").pipe(
          Effect.map(({ generation }) => generation),
          Effect.scoped,
        )).toBe(2)
        expect(yield* Ref.get(calls)).toBe(2)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("clears an interrupted startup so a later request can retry", async () => {
    const program = Effect.gen(function* () {
      const calls = yield* Ref.make(0)
      const entered = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      const setup = yield* makeSetup({
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
          .acquireSessionRequest(setup.request("interrupted"), "first")
          .pipe(Effect.scoped, Effect.fork)
        yield* Deferred.await(entered)
        yield* Fiber.interrupt(first)
        expect(
          yield* runtime.acquireSessionRequest(setup.request("interrupted"), "second").pipe(
            Effect.map(({ generation }) => generation),
            Effect.scoped,
          ),
        ).toBe(2)
        expect(yield* Ref.get(calls)).toBe(2)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("runs runtime finalizers before publishing retirement", async () => {
    const program = Effect.gen(function* () {
      const finalized = yield* Ref.make(false)
      const setup = yield* makeSetup({
        factory: {
          createSession: ({ scope }) =>
            Scope.addFinalizer(scope, Ref.set(finalized, true)).pipe(Effect.as(idleSession)),
        },
      })
      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* Effect.scoped(runtime.acquireSessionRequest(setup.request("finalize"), "use"))
        yield* runtime.dispose("finalize")
        expect(yield* Ref.get(finalized)).toBe(true)
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(setup.layer))
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
              subscribe: Stream.concat(
                Stream.succeed({ _tag: "Working" as const, workerCount: 1 }),
                Stream.never,
              ),
            },
          },
        }
        const setup = yield* makeSetup({
          factory: {
            createSession: ({ scope }) =>
              Scope.addFinalizer(scope, Ref.set(finalized, true)).pipe(Effect.as(workingSession)),
          },
        })

        yield* Effect.gen(function* () {
          const runtime = yield* AgentRuntime
          yield* Effect.scoped(
            runtime.acquireSessionRequest(setup.request("working-shutdown"), "start"),
          )
        }).pipe(Effect.provide(setup.layer))

        return yield* Ref.get(finalized)
      }),
    )

    expect(finalized).toBe(true)
  })

  it("excludes new admission while deletion drains and finalizes the loaded runtime", async () => {
    const program = Effect.gen(function* () {
      const useEntered = yield* Deferred.make<void>()
      const releaseUse = yield* Deferred.make<void>()
      const removalEntered = yield* Deferred.make<void>()
      const allowRemoval = yield* Deferred.make<void>()
      const finalized = yield* Ref.make(false)
      const finalizedBeforeRemoval = yield* Ref.make(false)
      const setup = yield* makeSetup({
        factory: {
          createSession: ({ scope }) =>
            Scope.addFinalizer(scope, Ref.set(finalized, true)).pipe(Effect.as(idleSession)),
        },
        storedSessionIds: ["delete"],
      })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const inFlight = yield* runtime
          .acquireSession("delete", "in-flight")
          .pipe(
            Effect.zipRight(Deferred.succeed(useEntered, undefined)),
            Effect.zipRight(Deferred.await(releaseUse)),
            Effect.scoped,
            Effect.fork,
          )
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
          Effect.scoped(runtime.acquireSession("delete", "too-late")),
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
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })

  it("keeps an accepted deletion owned after the requesting caller is interrupted", async () => {
    const program = Effect.gen(function* () {
      const removalEntered = yield* Deferred.make<void>()
      const allowRemoval = yield* Deferred.make<void>()
      const removalFinished = yield* Deferred.make<void>()
      const setup = yield* makeSetup({
        factory: { createSession: () => Effect.succeed(idleSession) },
        storedSessionIds: ["interrupt-delete"],
      })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        yield* Effect.scoped(runtime.acquireSession("interrupt-delete", "initial"))
        const caller = yield* runtime
          .deleteSession(
            "interrupt-delete",
            Deferred.succeed(removalEntered, undefined).pipe(
              Effect.zipRight(Deferred.await(allowRemoval)),
              Effect.zipRight(Deferred.succeed(removalFinished, undefined)),
            ),
          )
          .pipe(Effect.fork)
        yield* Deferred.await(removalEntered)
        yield* Fiber.interrupt(caller)
        yield* Deferred.succeed(allowRemoval, undefined)
        yield* Deferred.await(removalFinished)
        expect(yield* residentCount(runtime)).toBe(0)
      }).pipe(Effect.provide(setup.layer))
    })
    await Effect.runPromise(program)
  })
})
