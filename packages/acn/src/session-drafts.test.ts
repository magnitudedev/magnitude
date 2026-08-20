import { mkdtemp } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, it } from "vitest"
import { Deferred, Effect, Either, Fiber, Layer, Option, Ref, Scope, Stream } from "effect"
import type { CodingAgentSession } from "@magnitudedev/agent"
import { MagnitudeStorage, type StoredSessionMeta } from "@magnitudedev/storage"
import { DirectoryPathSchema, type DirectoryPath } from "@magnitudedev/acn-protocol"
import { AgentRuntime, type AgentRuntimeApi, type RuntimeStartRequest } from "./agent-runtime"
import { SessionDrafts, SessionDraftsLive } from "./session-drafts"
import { makeTestStorageLayer, testFileSystemManagerLayer } from "./session-test-support"
import {
  normalizeSessionRuntimeOptions,
  SessionRuntimeOptionsStore,
  type SessionRuntimeOptions,
  type SessionRuntimeOptionsStoreApi,
} from "./session-runtime-options"
import type { RuntimeEntry } from "./session-types"

const unusedSession = {
  on: { restoreQueuedMessages: Stream.never },
  state: {
    work: {
      get: () => Effect.succeed({ _tag: "Quiescent" as const, workerCount: 0 }),
      subscribe: Stream.succeed({ _tag: "Quiescent" as const, workerCount: 0 }),
    },
    turn: { getFork: () => Effect.die("unused"), subscribeFork: () => Stream.never },
    agentStatus: { get: () => Effect.die("unused"), subscribe: Stream.never },
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
} satisfies CodingAgentSession

const makeMeta = (
  sessionId: string,
  cwd: DirectoryPath,
  visibility: StoredSessionMeta["visibility"],
): StoredSessionMeta => {
  const now = new Date().toISOString()
  return {
    sessionId,
    archived: false,
    pinnedAt: Option.none(),
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

const makeSetup = Effect.gen(function* () {
  const root = yield* Effect.promise(() => mkdtemp(join(tmpdir(), "magnitude-drafts-")))
  const cwd = DirectoryPathSchema.make(root)
  const storage = yield* MagnitudeStorage.pipe(Effect.provide(makeTestStorageLayer(root)))
  const live = yield* Ref.make(new Map<string, RuntimeEntry>())
  const destroyed = yield* Ref.make<string[]>([])
  const starts = yield* Ref.make(0)
  const startRequests = yield* Ref.make<RuntimeStartRequest[]>([])
  const options = yield* Ref.make(new Map<string, SessionRuntimeOptions>())
  const serialize = yield* Effect.makeSemaphore(1)
  const startEntered = yield* Deferred.make<void>()
  const startDelay = yield* Ref.make<Deferred.Deferred<void> | null>(null)

  const makeEntry = Effect.fn("test.draft-entry")(function* (request: RuntimeStartRequest) {
    const scope = yield* Scope.make()
    return {
      id: request.sessionId,
      title: "New Chat",
      cwd: request.cwd,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      scratchpadPath: `/tmp/${request.sessionId}/scratchpad`,
      session: unusedSession,
      scope,
    } satisfies RuntimeEntry
  })

  const withSessionRequest: AgentRuntimeApi["withSessionRequest"] = (
    request,
    _label,
    use,
  ) =>
    serialize.withPermits(1)(
      Effect.gen(function* () {
        yield* Ref.update(startRequests, (all) => [...all, request])
        let entry = (yield* Ref.get(live)).get(request.sessionId)
        if (!entry) {
          yield* Deferred.succeed(startEntered, undefined)
          const delay = yield* Ref.get(startDelay)
          if (delay) yield* Deferred.await(delay)
          yield* Ref.update(starts, (count) => count + 1)
          entry = yield* makeEntry(request)
          yield* Ref.update(live, (all) => new Map(all).set(request.sessionId, entry!))
          yield* storage.sessions.writeMeta(
            request.sessionId,
            makeMeta(request.sessionId, request.cwd, request.visibility),
          ).pipe(Effect.orDie)
        }
        return yield* use(entry, 1)
      }),
    )

  const runtime: AgentRuntimeApi = {
    withSession: (sessionId, _label, use) =>
      Ref.get(live).pipe(
        Effect.flatMap((all) => {
          const entry = all.get(sessionId)
          return entry ? use(entry, 1) : Effect.die("missing fake session")
        }),
      ),
    withSessionRequest,
    tryWithResident: () => Effect.succeed(Option.none()),
    tryWithBusyResident: () => Effect.succeed(Option.none()),
    residentSessions: Effect.succeed([]),
    dispose: (sessionId) =>
      Ref.update(live, (all) => {
        const next = new Map(all)
        next.delete(sessionId)
        return next
      }),
    deleteSession: (sessionId, remove) =>
      Ref.update(destroyed, (all) => [...all, sessionId]).pipe(Effect.zipRight(remove)),
    registerRetirementObserver: () => Effect.succeed(Effect.void),
    changes: Stream.never,
  }

  const optionStore: SessionRuntimeOptionsStoreApi = {
    normalize: normalizeSessionRuntimeOptions,
    read: (sessionId) => Ref.get(options).pipe(Effect.map((all) => all.get(sessionId) ?? null)),
    write: (sessionId, value) => Ref.update(options, (all) => new Map(all).set(sessionId, value)),
  }

  return {
    cwd,
    storage,
    layer: SessionDraftsLive.pipe(
      Layer.provide(
        Layer.mergeAll(
          Layer.succeed(AgentRuntime, runtime),
          Layer.succeed(MagnitudeStorage, storage),
          testFileSystemManagerLayer,
          Layer.succeed(SessionRuntimeOptionsStore, optionStore),
        ),
      ),
    ),
    runtime,
    refs: { destroyed, starts, startRequests, startEntered, startDelay },
  }
})

describe("SessionDrafts", () => {
  it("deduplicates a key while AgentRuntime single-flights its generation", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const results = yield* Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        return yield* Effect.all(
          [
            drafts.preload({ cwd: setup.cwd, ownerId: "owner" }),
            drafts.preload({ cwd: setup.cwd, ownerId: "owner" }),
          ],
          { concurrency: "unbounded" },
        )
      }).pipe(Effect.provide(setup.layer))
      expect(results[0].sessionId).toBe(results[1].sessionId)
      expect(yield* Ref.get(setup.refs.starts)).toBe(1)
    })
    await Effect.runPromise(program)
  })

  it("releases an empty unclaimed draft", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const id = yield* Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        const preload = yield* drafts.preload({ cwd: setup.cwd, ownerId: "owner" })
        yield* drafts.release({ cwd: setup.cwd, sessionId: preload.sessionId, ownerId: "owner" })
        return preload.sessionId
      }).pipe(Effect.provide(setup.layer))
      expect(yield* Ref.get(setup.refs.destroyed)).toEqual([id])
      expect(yield* setup.storage.sessions.readMeta(id).pipe(Effect.orDie)).toBeNull()
    })
    await Effect.runPromise(program)
  })

  it("claims before awaiting startup so a racing release cannot destroy it", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const releaseStart = yield* Deferred.make<void>()
      yield* Ref.set(setup.refs.startDelay, releaseStart)
      const id = yield* Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        const claiming = yield* drafts.claim({ cwd: setup.cwd, ownerId: "owner" }).pipe(Effect.fork)
        yield* Deferred.await(setup.refs.startEntered)
        const inFlightId = (yield* Ref.get(setup.refs.startRequests))[0]?.sessionId ?? ""
        yield* drafts.release({ cwd: setup.cwd, sessionId: inFlightId, ownerId: "owner" })
        yield* Deferred.succeed(releaseStart, undefined)
        const claim = yield* claiming
        yield* drafts.promote(claim)
        return claim.sessionId
      }).pipe(Effect.provide(setup.layer))
      expect(yield* Ref.get(setup.refs.destroyed)).toEqual([])
      const meta = yield* setup.storage.sessions.readMeta(id).pipe(Effect.orDie)
      expect(meta?.visibility).toBe("visible")
    })
    await Effect.runPromise(program)
  })

  it("allows exactly one concurrent claimer", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const results = yield* Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        return yield* Effect.all(
          [
            Effect.either(drafts.claim({ cwd: setup.cwd, ownerId: "owner" })),
            Effect.either(drafts.claim({ cwd: setup.cwd, ownerId: "owner" })),
          ],
          { concurrency: "unbounded" },
        )
      }).pipe(Effect.provide(setup.layer))
      expect(results.filter(Either.isRight)).toHaveLength(1)
      expect(results.filter(Either.isLeft)).toHaveLength(1)
    })
    await Effect.runPromise(program)
  })

  it("restores a claim when its caller is interrupted during initialization", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const releaseStart = yield* Deferred.make<void>()
      yield* Ref.set(setup.refs.startDelay, releaseStart)
      const ids = yield* Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        const first = yield* drafts.claim({ cwd: setup.cwd, ownerId: "owner" }).pipe(Effect.fork)
        yield* Deferred.await(setup.refs.startEntered)
        yield* Fiber.interrupt(first)
        yield* Ref.set(setup.refs.startDelay, null)
        const firstId = (yield* Ref.get(setup.refs.startRequests))[0]?.sessionId ?? ""
        const second = yield* drafts.claim({ cwd: setup.cwd, ownerId: "owner" })
        return { firstId, secondId: second.sessionId }
      }).pipe(Effect.provide(setup.layer))
      expect(ids.secondId).toBe(ids.firstId)
    })
    await Effect.runPromise(program)
  })

  it("removes a preloading record when its caller is interrupted", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      const releaseStart = yield* Deferred.make<void>()
      yield* Ref.set(setup.refs.startDelay, releaseStart)
      const ids = yield* Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        const preload = yield* drafts.preload({ cwd: setup.cwd, ownerId: "owner" }).pipe(Effect.fork)
        yield* Deferred.await(setup.refs.startEntered)
        yield* Fiber.interrupt(preload)
        yield* Ref.set(setup.refs.startDelay, null)
        const firstId = (yield* Ref.get(setup.refs.startRequests))[0]?.sessionId ?? ""
        const next = yield* drafts.preload({ cwd: setup.cwd, ownerId: "owner" })
        return { firstId, nextId: next.sessionId }
      }).pipe(Effect.provide(setup.layer))
      expect(ids.nextId).not.toBe(ids.firstId)
      expect(yield* Ref.get(setup.refs.destroyed)).toContain(ids.firstId)
    })
    await Effect.runPromise(program)
  })

  it("preserves normalized options at the runtime boundary", async () => {
    const program = Effect.gen(function* () {
      const setup = yield* makeSetup
      yield* Effect.gen(function* () {
        const drafts = yield* SessionDrafts
        yield* drafts.preload({
          cwd: setup.cwd,
          ownerId: "owner",
          options: {
            disableShellSafeguards: true,
            disableCwdSafeguards: false,
            solo: true,
            headless: false,
          },
        })
      }).pipe(Effect.provide(setup.layer))
      const request = (yield* Ref.get(setup.refs.startRequests))[0]
      expect(request?.options).toEqual(
        normalizeSessionRuntimeOptions({
          disableShellSafeguards: true,
          disableCwdSafeguards: false,
          solo: true,
          headless: false,
        }),
      )
    })
    await Effect.runPromise(program)
  })
})
