import { describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Layer, Option, Ref, Scope, Stream } from "effect"
import type { AgentLifecycleState, CodingAgentSession, ForkTurnState } from "@magnitudedev/agent"
import type { StoredSessionMeta } from "@magnitudedev/storage"
import { SessionOperationFailed } from "@magnitudedev/protocol"
import { AgentFactory, type AgentFactoryApi } from "./agent-factory"
import { AgentRuntime, AgentRuntimeLive, type RuntimeStartRequest } from "./agent-runtime"
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

const unusedCodingAgentSession: CodingAgentSession = {
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
  send: () => Effect.die("unused test session send"),
  interrupt: () => Effect.die("unused test session interrupt"),
  refreshConfig: () => Effect.void,
  publishInitialTask: () => Effect.void,
  onEvent: Stream.never,
  onError: Stream.never,
  subscribeIntrospection: () => Stream.never,
}

const makeMeta = (
  sessionId: string,
  cwd: string,
  visibility: StoredSessionMeta["visibility"],
): StoredSessionMeta => {
  const now = new Date().toISOString()
  return {
    sessionId,
    created: now,
    updated: now,
    chatName: "Session",
    workingDirectory: cwd,
    visibility,
    initialVersion: "0.0.1",
    lastActiveVersion: "0.0.1",
    gitBranch: null,
    firstUserMessage: null,
    lastMessage: null,
    messageCount: 0,
  }
}

const makeLayer = (input: {
  readonly factory: AgentFactoryApi
  readonly storedSessions?: ReadonlyArray<StoredSessionMeta>
  readonly storedRuntimeOptions?: ReadonlyMap<string, SessionRuntimeOptions>
}) =>
  AgentRuntimeLive.pipe(
    Layer.provide(Layer.mergeAll(
      Layer.succeed(AgentFactory, input.factory),
      Layer.effect(
        SessionStore,
        Effect.gen(function* () {
          const metas = yield* Ref.make(new Map<string, StoredSessionMeta>())
          for (const meta of input.storedSessions ?? []) {
            yield* Ref.update(metas, (map) => new Map(map).set(meta.sessionId, meta))
          }

          return {
            createId: Effect.die("unused"),
            readMeta: (sessionId) =>
              Ref.get(metas).pipe(Effect.map((map) => map.get(sessionId) ?? null)),
            readProtocolMeta: () => Effect.die("unused"),
            promoteDraft: () => Effect.die("unused"),
            listDraftSessionIds: () => Effect.die("unused"),
            listProtocolMetas: () => Effect.die("unused"),
            listSessionCwds: () => Effect.die("unused"),
            deleteSessionFiles: () => Effect.die("unused"),
            validateCwd: (cwd) => Effect.succeed(cwd),
            getScratchpadPath: (sessionId) => Effect.succeed(`/tmp/${sessionId}/scratchpad`),
            getExecutionContext: () => Effect.die("unused"),

          } satisfies SessionStoreApi
        }),
      ),
      Layer.effect(
        SessionRuntimeOptionsStore,
        Effect.gen(function* () {
          const runtimeOptions = yield* Ref.make(new Map(input.storedRuntimeOptions ?? []))
          const api: SessionRuntimeOptionsStoreApi = {
            normalize: normalizeSessionRuntimeOptions,
            read: (sessionId) =>
              Ref.get(runtimeOptions).pipe(
                Effect.map((map) => map.get(sessionId) ?? null),
              ),
            write: (sessionId, options) =>
              Ref.update(runtimeOptions, (map) => new Map(map).set(sessionId, options)),
          }
          return api
        }),
      ),
    )),
  )

describe("AgentRuntime", () => {
  it("single-flights concurrent getOrStart calls for the same session", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const createCalls = yield* Ref.make(0)
      const entered = yield* Deferred.make<void>()
      const release = yield* Deferred.make<void>()
      const factory: AgentFactoryApi = {
        createSession: () =>
          Effect.gen(function* () {
            yield* Ref.update(createCalls, (count) => count + 1)
            yield* Deferred.succeed(entered, undefined)
            yield* Deferred.await(release)
            return unusedCodingAgentSession
          }),
      }
      const layer = makeLayer({ factory })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const request: RuntimeStartRequest = {
          sessionId: "session-a",
          cwd: "/repo",
          options: normalizeSessionRuntimeOptions(),
          visibility: "visible",
          scope: Option.none(),
        }

        const first = yield* Effect.fork(runtime.getOrStart(request))
        yield* Deferred.await(entered)
        const second = yield* Effect.fork(runtime.getOrStart(request))
        yield* Deferred.succeed(release, undefined)

        const firstEntry = yield* Fiber.join(first)
        const secondEntry = yield* Fiber.join(second)
        const calls = yield* Ref.get(createCalls)
        const count = yield* runtime.count

        expect(firstEntry.id).toBe("session-a")
        expect(secondEntry.id).toBe("session-a")
        expect(firstEntry.session).toBe(secondEntry.session)
        expect(calls).toBe(1)
        expect(count).toBe(1)
      }).pipe(Effect.provide(layer))
    }))
  })

  it("single-flights concurrent requireOrStart calls for a persisted session", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const createCalls = yield* Ref.make(0)
      const entered = yield* Deferred.make<void>()
      const release = yield* Deferred.make<void>()
      const factory: AgentFactoryApi = {
        createSession: () =>
          Effect.gen(function* () {
            yield* Ref.update(createCalls, (count) => count + 1)
            yield* Deferred.succeed(entered, undefined)
            yield* Deferred.await(release)
            return unusedCodingAgentSession
          }),
      }
      const layer = makeLayer({
        factory,
        storedSessions: [makeMeta("session-b", "/repo", "visible")],
        storedRuntimeOptions: new Map([[
          "session-b",
          normalizeSessionRuntimeOptions({
            disableShellSafeguards: false,
            disableCwdSafeguards: false,
            solo: true,
            headless: true,
          }),
        ]]),
      })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const first = yield* Effect.fork(runtime.requireOrStart("session-b"))
        yield* Deferred.await(entered)
        const second = yield* Effect.fork(runtime.requireOrStart("session-b"))
        yield* Deferred.succeed(release, undefined)

        const firstEntry = yield* Fiber.join(first)
        const secondEntry = yield* Fiber.join(second)
        const calls = yield* Ref.get(createCalls)
        const count = yield* runtime.count

        expect(firstEntry.id).toBe("session-b")
        expect(secondEntry.id).toBe("session-b")
        expect(firstEntry.session).toBe(secondEntry.session)
        expect(calls).toBe(1)
        expect(count).toBe(1)
      }).pipe(Effect.provide(layer))
    }))
  })

  it("starts persisted sessions with stored runtime options", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const storedOptions = normalizeSessionRuntimeOptions({
        disableShellSafeguards: true,
        disableCwdSafeguards: true,
        atifPath: "/tmp/atif.md",
        solo: true,
        systemPromptOverride: "system",
        headless: true,
      })
      const seenOptions = yield* Ref.make<SessionRuntimeOptions | null>(null)
      const factory: AgentFactoryApi = {
        createSession: (input) =>
          Ref.set(seenOptions, input.options).pipe(
            Effect.as(unusedCodingAgentSession),
          ),
      }
      const layer = makeLayer({
        factory,
        storedSessions: [makeMeta("session-options", "/repo", "visible")],
        storedRuntimeOptions: new Map([["session-options", storedOptions]]),
      })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const entry = yield* runtime.requireOrStart("session-options")
        const options = yield* Ref.get(seenOptions)

        expect(entry.id).toBe("session-options")
        expect(options).toEqual(storedOptions)
      }).pipe(Effect.provide(layer))
    }))
  })

  it("clears failed starts so later calls can retry", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const createCalls = yield* Ref.make(0)
      const failEntered = yield* Deferred.make<void>()
      const failRelease = yield* Deferred.make<void>()
      const failed = new SessionOperationFailed({
        operation: "test-start",
        reason: "boom",
      })
      const factory: AgentFactoryApi = {
        createSession: () =>
          Effect.gen(function* () {
            const call = yield* Ref.updateAndGet(createCalls, (count) => count + 1)
            if (call === 1) {
              yield* Deferred.succeed(failEntered, undefined)
              yield* Deferred.await(failRelease)
              return yield* failed
            }
            return unusedCodingAgentSession
          }),
      }
      const layer = makeLayer({ factory })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const request: RuntimeStartRequest = {
          sessionId: "session-c",
          cwd: "/repo",
          options: normalizeSessionRuntimeOptions(),
          visibility: "visible",
          scope: Option.none(),
        }

        const first = yield* Effect.fork(Effect.flip(runtime.getOrStart(request)))
        yield* Deferred.await(failEntered)
        const second = yield* Effect.fork(Effect.flip(runtime.getOrStart(request)))
        yield* Deferred.succeed(failRelease, undefined)

        const firstError = yield* Fiber.join(first)
        const secondError = yield* Fiber.join(second)
        const retried = yield* runtime.getOrStart(request)
        const calls = yield* Ref.get(createCalls)

        expect(firstError).toStrictEqual(failed)
        expect(secondError).toStrictEqual(failed)
        expect(retried.id).toBe("session-c")
        expect(calls).toBe(2)
      }).pipe(Effect.provide(layer))
    }))
  })

  it("removes disposed sessions before waiting on scope finalizers", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const createCalls = yield* Ref.make(0)
      const closeStarted = yield* Deferred.make<void>()
      const releaseClose = yield* Deferred.make<void>()
      const firstSession: CodingAgentSession = { ...unusedCodingAgentSession }
      const secondSession: CodingAgentSession = { ...unusedCodingAgentSession }
      const factory: AgentFactoryApi = {
        createSession: (input) =>
          Effect.gen(function* () {
            const call = yield* Ref.updateAndGet(createCalls, (count) => count + 1)
            if (call === 1) {
              yield* Scope.addFinalizer(
                input.scope,
                Deferred.succeed(closeStarted, undefined).pipe(
                  Effect.zipRight(Deferred.await(releaseClose)),
                ),
              )
            }
            return call === 1 ? firstSession : secondSession
          }),
      }
      const layer = makeLayer({
        factory,
        storedSessions: [makeMeta("session-d", "/repo", "visible")],
      })

      yield* Effect.gen(function* () {
        const runtime = yield* AgentRuntime
        const request: RuntimeStartRequest = {
          sessionId: "session-d",
          cwd: "/repo",
          options: normalizeSessionRuntimeOptions(),
          visibility: "visible",
          scope: Option.none(),
        }

        const first = yield* runtime.getOrStart(request)
        const disposing = yield* Effect.fork(runtime.dispose("session-d"))
        yield* Deferred.await(closeStarted)

        const restarted = yield* runtime.requireOrStart("session-d")
        const calls = yield* Ref.get(createCalls)

        expect(first.session).toBe(firstSession)
        expect(restarted.session).toBe(secondSession)
        expect(calls).toBe(2)

        yield* Deferred.succeed(releaseClose, undefined)
        yield* Fiber.join(disposing)
      }).pipe(Effect.provide(layer))
    }))
  })
})
