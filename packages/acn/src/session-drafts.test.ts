import { describe, expect, test } from "vitest"
import { Duration, Effect, Either, Layer, Ref, Scope, Stream } from "effect"
import type { CodingAgentSession } from "@magnitudedev/agent"
import type { StoredSessionMeta } from "@magnitudedev/storage"
import { SessionNotFound, SessionOperationFailed } from "@magnitudedev/protocol"
import { AgentRuntime, type AgentRuntimeApi, type RuntimeStartRequest } from "./agent-runtime"
import { SessionDrafts, SessionDraftsLive } from "./session-drafts"
import { SessionStore, type SessionStoreApi } from "./session-store"
import {
  normalizeSessionRuntimeOptions,
  SessionRuntimeOptionsStore,
  type SessionRuntimeOptions,
  type SessionRuntimeOptionsStoreApi,
} from "./session-runtime-options"
import type { RuntimeEntry } from "./session-types"

const makeMeta = (sessionId: string, cwd: string, visibility: StoredSessionMeta["visibility"]): StoredSessionMeta => {
  const now = new Date().toISOString()
  return {
    sessionId,
    created: now,
    updated: now,
    chatName: "New Chat",
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

const unusedCodingAgentSession: CodingAgentSession = {
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
  subscribeIntrospection: () => Stream.never,
}

const makeEntry = (sessionId: string, cwd: string): Effect.Effect<RuntimeEntry> =>
  Effect.gen(function* () {
    const scope = yield* Scope.make()
    return {
      id: sessionId,
      title: "New Chat",
      cwd,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      scratchpadPath: `/tmp/${sessionId}/scratchpad`,
      session: unusedCodingAgentSession,
      scope,
    }
  })

const makeTestLayer = (overrides?: {
  readonly getOrStart?: AgentRuntimeApi["getOrStart"]
}) => Effect.gen(function* () {
  const metas = yield* Ref.make(new Map<string, StoredSessionMeta>())
  const live = yield* Ref.make(new Map<string, RuntimeEntry>())
  const destroyed = yield* Ref.make<string[]>([])
  const startCount = yield* Ref.make(0)
  const idCounter = yield* Ref.make(0)
  const runtimeOptions = yield* Ref.make(new Map<string, SessionRuntimeOptions>())
  const startRequests = yield* Ref.make<RuntimeStartRequest[]>([])

  const store: SessionStoreApi = {
    createId: Ref.modify(idCounter, (count) => [`draft-${count}`, count + 1] as const),
    readMeta: (sessionId) => Ref.get(metas).pipe(Effect.map((map) => map.get(sessionId) ?? null)),
    readProtocolMeta: (sessionId) =>
      Ref.get(metas).pipe(
        Effect.map((map) => {
          const meta = map.get(sessionId)
          if (!meta) return null
          return {
            sessionId,
            title: meta.chatName,
            cwd: meta.workingDirectory,
            createdAt: Date.parse(meta.created),
            updatedAt: Date.parse(meta.updated),
            messageCount: meta.messageCount,
            lastMessage: meta.lastMessage,
          }
        }),
      ),
    promoteDraft: (sessionId) =>
      Ref.modify(metas, (map) => {
        const current = map.get(sessionId)
        if (!current) throw new Error(`missing meta ${sessionId}`)
        const nextMeta = { ...current, visibility: "visible" as const }
        const next = new Map(map)
        next.set(sessionId, nextMeta)
        return [nextMeta, next] as const
      }),
    listDraftSessionIds: () =>
      Ref.get(metas).pipe(
        Effect.map((map) => [...map.values()].filter((meta) => meta.visibility === "draft").map((meta) => meta.sessionId)),
      ),
    listProtocolMetas: () => Effect.die("unused"),
    listSessionCwds: () => Effect.die("unused"),
    deleteSessionFiles: (sessionId) =>
      Effect.gen(function* () {
        yield* Ref.update(destroyed, (ids) => [...ids, sessionId])
        yield* Ref.update(metas, (map) => {
          const next = new Map(map)
          next.delete(sessionId)
          return next
        })
      }),
    validateCwd: (cwd) => Effect.succeed(cwd),
    getScratchpadPath: (sessionId) => Effect.succeed(`/tmp/${sessionId}/scratchpad`),
    getExecutionContext: () => Effect.die("unused"),

  }

  const runtime: AgentRuntimeApi = {
    getLive: (sessionId) => Ref.get(live).pipe(Effect.map((map) => map.get(sessionId) ?? null)),
    getAllEntries: () => Ref.get(live).pipe(Effect.map((map) => [...map.values()])),
    getOrStart: overrides?.getOrStart ?? ((request) =>
      Effect.gen(function* () {
        yield* Ref.update(startRequests, (requests) => [...requests, request])
        yield* Effect.sleep("10 millis")
        yield* Ref.update(startCount, (count) => count + 1)
        const entry = yield* makeEntry(request.sessionId, request.cwd)
        yield* Ref.update(live, (map) => new Map(map).set(request.sessionId, entry))
        yield* Ref.update(metas, (map) =>
          new Map(map).set(request.sessionId, makeMeta(request.sessionId, request.cwd, request.visibility))
        )
        return entry
      })),
    requireOrStart: (sessionId) =>
      Effect.flatMap(
        Ref.get(live),
        (map) => {
          const entry = map.get(sessionId)
          return entry ? Effect.succeed(entry) : Effect.fail(new SessionNotFound({ sessionId }))
        },
      ),
    dispose: (sessionId) =>
      Ref.update(live, (map) => {
        const next = new Map(map)
        next.delete(sessionId)
        return next
      }),
    disposeAll: () => Ref.set(live, new Map()),
    touchEntry: () => Effect.void,
    retainEntry: () => Effect.void,
    releaseEntry: () => Effect.void,
    evictIdleSessions: () => Effect.void,
    hasActiveWork: Effect.succeed(false),
    count: Ref.get(live).pipe(Effect.map((map) => map.size)),
    changes: Stream.never,
  }

  const runtimeOptionsStore: SessionRuntimeOptionsStoreApi = {
    normalize: normalizeSessionRuntimeOptions,
    read: (sessionId) =>
      Ref.get(runtimeOptions).pipe(
        Effect.map((map) => map.get(sessionId) ?? null),
      ),
    write: (sessionId, options) =>
      Ref.update(runtimeOptions, (map) => new Map(map).set(sessionId, options)),
  }

  return {
    layer: SessionDraftsLive.pipe(
      Layer.provide(Layer.mergeAll(
        Layer.succeed(SessionStore, store),
        Layer.succeed(AgentRuntime, runtime),
        Layer.succeed(SessionRuntimeOptionsStore, runtimeOptionsStore),
      )),
    ),
    refs: { destroyed, startCount, metas, startRequests },
  }
})

describe("SessionDrafts", () => {
  test("dedupes concurrent preload calls for the same key", async () => {
    const { layer, refs } = await Effect.runPromise(makeTestLayer())
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        return yield* Effect.all([
          drafts.preload({ cwd: "/repo", ownerId: "owner" }),
          drafts.preload({ cwd: "/repo", ownerId: "owner" }),
        ], { concurrency: "unbounded" })
      }).pipe(Effect.provide(layer)),
    )

    const startCount = await Effect.runPromise(Ref.get(refs.startCount))
    expect(result[0].sessionId).toBe(result[1].sessionId)
    expect(startCount).toBe(1)
  })

  test("release destroys an unclaimed ready draft", async () => {
    const { layer, refs } = await Effect.runPromise(makeTestLayer())
    const sessionId = await Effect.runPromise(
      Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        const preload = yield* drafts.preload({ cwd: "/repo", ownerId: "owner" })
        yield* drafts.release({ cwd: "/repo", ownerId: "owner" })
        return preload.sessionId
      }).pipe(Effect.provide(layer)),
    )

    const destroyed = await Effect.runPromise(Ref.get(refs.destroyed))
    const metas = await Effect.runPromise(Ref.get(refs.metas))
    expect(destroyed).toEqual([sessionId])
    expect(metas.has(sessionId)).toBe(false)
  })

  test("preload starts drafts with normalized options", async () => {
    const { layer, refs } = await Effect.runPromise(makeTestLayer())
    await Effect.runPromise(
      Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        yield* drafts.preload({
          cwd: "/repo",
          ownerId: "owner",
          options: {
            disableShellSafeguards: true,
            disableCwdSafeguards: false,
            solo: true,
            systemPromptOverride: "system",
            headless: false,
          },
        })
      }).pipe(Effect.provide(layer)),
    )

    const requests = await Effect.runPromise(Ref.get(refs.startRequests))
    expect(requests).toHaveLength(1)
    const request = requests[0]
    expect(request).toBeDefined()
    if (!request) throw new Error("missing runtime start request")
    expect(request.options).toEqual(normalizeSessionRuntimeOptions({
      disableShellSafeguards: true,
      disableCwdSafeguards: false,
      solo: true,
      systemPromptOverride: "system",
      headless: false,
    }))
  })

  test("claim plus promote makes the draft visible and removes it from cleanup", async () => {
    const { layer, refs } = await Effect.runPromise(makeTestLayer())
    const sessionId = await Effect.runPromise(
      Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        const preload = yield* drafts.preload({ cwd: "/repo", ownerId: "owner" })
        const claim = yield* drafts.claim({ cwd: "/repo", ownerId: "owner" })
        yield* drafts.promote(claim)
        yield* drafts.release({ cwd: "/repo", ownerId: "owner" })
        return preload.sessionId
      }).pipe(Effect.provide(layer)),
    )

    const destroyed = await Effect.runPromise(Ref.get(refs.destroyed))
    const metas = await Effect.runPromise(Ref.get(refs.metas))
    expect(destroyed).toEqual([])
    expect(metas.get(sessionId)?.visibility).toBe("visible")
  })

  test("release during in-flight claim does not destroy the draft (TOCTOU race)", async () => {
    const { layer, refs } = await Effect.runPromise(makeTestLayer())
    const sessionId = await Effect.runPromise(
      Effect.gen(function* () {
        const drafts = yield* SessionDrafts

        // Fork claim — it will suspend on awaitEntry for ~10ms (test layer's getOrStart sleep).
        // claim marks "claiming" atomically before awaiting, so the entry is "claiming"
        // during the entire suspension window.
        const claimFiber = yield* Effect.fork(
          drafts.claim({ cwd: "/repo", ownerId: "owner" }),
        )

        // Give claim time to run ensureEntry + mark "claiming" + start awaiting
        yield* Effect.sleep("5 millis")

        // Now release arrives — this is the race window. With the fix, release
        // sees state "claiming" and bails. Without the fix, release sees
        // "preloading"/"ready" and destroys the backing session.
        yield* drafts.release({ cwd: "/repo", ownerId: "owner" })

        // Claim should still succeed despite the racing release
        const claim = yield* claimFiber
        yield* drafts.promote(claim)
        return claim.sessionId
      }).pipe(Effect.provide(layer)),
    )

    const destroyed = await Effect.runPromise(Ref.get(refs.destroyed))
    const metas = await Effect.runPromise(Ref.get(refs.metas))
    // The backing session must NOT have been destroyed by the racing release
    expect(destroyed).toEqual([])
    expect(metas.get(sessionId)?.visibility).toBe("visible")
  })

  test("release during preloading entry does not destroy a concurrent claim", async () => {
    const { layer, refs } = await Effect.runPromise(makeTestLayer())
    await Effect.runPromise(
      Effect.gen(function* () {
        const drafts = yield* SessionDrafts

        // Fork preload to create the entry in "preloading" state
        const preloadFiber = yield* Effect.fork(
          drafts.preload({ cwd: "/repo", ownerId: "owner" }),
        )
        yield* Effect.sleep("5 millis")

        // Fork claim — it will find the existing entry, mark "claiming", then await
        const claimFiber = yield* Effect.fork(
          drafts.claim({ cwd: "/repo", ownerId: "owner" }),
        )
        yield* Effect.sleep("5 millis")

        // Release arrives while claim is suspended on awaitEntry
        yield* drafts.release({ cwd: "/repo", ownerId: "owner" })

        // Claim should succeed
        const claim = yield* claimFiber
        yield* drafts.promote(claim)

        // Clean up preload fiber (may have already resolved or will resolve to the same entry)
        yield* Effect.zipRight(preloadFiber, Effect.void)
      }).pipe(Effect.provide(layer)),
    )

    const destroyed = await Effect.runPromise(Ref.get(refs.destroyed))
    expect(destroyed).toEqual([])
  })

  test("two concurrent claims on the same key — first wins, second fails", async () => {
    const { layer } = await Effect.runPromise(makeTestLayer())
    const results = await Effect.runPromise(
      Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        return yield* Effect.all([
          Effect.either(drafts.claim({ cwd: "/repo", ownerId: "owner" })),
          Effect.either(drafts.claim({ cwd: "/repo", ownerId: "owner" })),
        ], { concurrency: "unbounded" })
      }).pipe(Effect.provide(layer)),
    )

    const wins = results.filter((r) => Either.isRight(r))
    const fails = results.filter((r) => Either.isLeft(r))
    expect(wins).toHaveLength(1)
    expect(fails).toHaveLength(1)
  })

  test("claim receives startup failure after it wins the claiming race", async () => {
    const startupError = new SessionOperationFailed({
      operation: "start draft",
      reason: "boom",
    })
    const { layer } = await Effect.runPromise(makeTestLayer({
      getOrStart: () =>
        Effect.sleep("10 millis").pipe(
          Effect.zipRight(Effect.fail(startupError)),
        ),
    }))

    const result = await Effect.runPromise(
      Effect.either(
        Effect.gen(function* () {
          const drafts = yield* SessionDrafts
          return yield* drafts.claim({ cwd: "/repo", ownerId: "owner" })
        }).pipe(
          Effect.provide(layer),
          Effect.timeoutFail({
            duration: Duration.millis(250),
            onTimeout: () => new SessionOperationFailed({
              operation: "claim draft",
              reason: "claim hung waiting for startup failure",
            }),
          }),
        ),
      ),
    )

    expect(Either.isLeft(result)).toBe(true)
    if (Either.isLeft(result)) {
      expect(result.left).toMatchObject({
        _tag: "SessionOperationFailed",
        operation: startupError.operation,
        reason: startupError.reason,
      })
    }
  })
})
