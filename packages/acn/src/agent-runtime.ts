import { Context, Deferred, Effect, Layer, Ref, Scope, Exit, PubSub, Stream, Option } from "effect"
import { DEFAULT_CHAT_NAME } from "@magnitudedev/agent"
import {
  SessionNotFound,
  type SessionError,
} from "@magnitudedev/protocol"
import type { StoredSessionMeta } from "@magnitudedev/storage"
import { AgentFactory } from "./agent-factory"
import { SessionStore } from "./session-store"
import {
  SessionRuntimeOptionsStore,
  normalizeSessionRuntimeOptions,
  type SessionRuntimeOptions,
} from "./session-runtime-options"
import type { RuntimeEntry } from "./session-types"
import type { CloseableScope } from "effect/Scope"

export interface RuntimeStartRequest {
  readonly sessionId: string
  readonly cwd: string
  readonly options: SessionRuntimeOptions
  readonly visibility: StoredSessionMeta["visibility"]
  readonly scope: Option.Option<CloseableScope>
}

export interface AgentRuntimeApi {
  readonly getLive: (sessionId: string) => Effect.Effect<RuntimeEntry | null>
  readonly getAllEntries: () => Effect.Effect<ReadonlyArray<RuntimeEntry>>
  readonly getOrStart: (request: RuntimeStartRequest) => Effect.Effect<RuntimeEntry, SessionError>
  readonly requireOrStart: (sessionId: string) => Effect.Effect<RuntimeEntry, SessionError>
  readonly dispose: (sessionId: string) => Effect.Effect<void>
  readonly disposeAll: () => Effect.Effect<void>
  readonly touchEntry: (sessionId: string) => Effect.Effect<void>
  readonly retainEntry: (sessionId: string) => Effect.Effect<void>
  readonly releaseEntry: (sessionId: string) => Effect.Effect<void>
  readonly evictIdleSessions: () => Effect.Effect<void>
  readonly hasActiveWork: Effect.Effect<boolean>
  readonly count: Effect.Effect<number>
  readonly changes: Stream.Stream<void>
}

const SESSION_EVICT_GRACE_MS = 120_000

export class AgentRuntime extends Context.Tag("AgentRuntime")<
  AgentRuntime,
  AgentRuntimeApi
>() {}

type RuntimeStartDeferred = Deferred.Deferred<RuntimeEntry, SessionError>

type RuntimeStartClaim =
  | { readonly _tag: "owner"; readonly deferred: RuntimeStartDeferred }
  | { readonly _tag: "joiner"; readonly deferred: RuntimeStartDeferred }

export const AgentRuntimeLive: Layer.Layer<AgentRuntime, never, AgentFactory | SessionStore | SessionRuntimeOptionsStore> =
  Layer.scoped(
    AgentRuntime,
    Effect.gen(function* () {
      const factory = yield* AgentFactory
      const store = yield* SessionStore
      const runtimeOptions = yield* SessionRuntimeOptionsStore
      const entries = yield* Ref.make(new Map<string, RuntimeEntry>())
      const retains = yield* Ref.make(new Map<string, number>())
      const starts = yield* Ref.make(new Map<string, RuntimeStartDeferred>())
      const changes = yield* PubSub.unbounded<void>()

      const publishChange = PubSub.publish(changes, undefined).pipe(Effect.asVoid)

      const getLive = Effect.fn("acn.agent-runtime.get-live")(function* (sessionId: string) {
        return (yield* Ref.get(entries)).get(sessionId) ?? null
      })

      const touchEntryNow = (sessionId: string) =>
        Ref.modify(entries, (map): readonly [RuntimeEntry | null, Map<string, RuntimeEntry>] => {
          const entry = map.get(sessionId)
          if (!entry) return [null, map]

          const touched = { ...entry, updatedAt: Date.now() }
          return [touched, new Map(map).set(sessionId, touched)]
        })

      const touchEntry = Effect.fn("acn.agent-runtime.touch-entry")(function* (sessionId: string) {
        yield* touchEntryNow(sessionId)
      })

      const retainEntry = Effect.fn("acn.agent-runtime.retain-entry")(function* (sessionId: string) {
        const entry = yield* getLive(sessionId)
        if (!entry) return

        yield* Ref.update(retains, (map) => {
          const next = new Map(map)
          next.set(sessionId, (next.get(sessionId) ?? 0) + 1)
          return next
        })
        yield* touchEntry(sessionId)
      })

      const releaseEntry = Effect.fn("acn.agent-runtime.release-entry")(function* (sessionId: string) {
        yield* Ref.update(retains, (map) => {
          const current = map.get(sessionId) ?? 0
          if (current <= 0) return map

          const next = new Map(map)
          if (current === 1) next.delete(sessionId)
          else next.set(sessionId, current - 1)
          return next
        })
        yield* touchEntry(sessionId)
      })

      const claimStart = Effect.fn("acn.agent-runtime.claim-start")(function* (sessionId: string) {
        const deferred = yield* Deferred.make<RuntimeEntry, SessionError>()
        return yield* Ref.modify(starts, (map): readonly [RuntimeStartClaim, Map<string, RuntimeStartDeferred>] => {
          const current = map.get(sessionId)
          if (current) return [{ _tag: "joiner", deferred: current }, map]

          const next = new Map(map)
          next.set(sessionId, deferred)
          return [{ _tag: "owner", deferred }, next]
        })
      })

      const clearStart = (
        sessionId: string,
        deferred: RuntimeStartDeferred,
      ): Effect.Effect<void> =>
        Ref.update(starts, (map) => {
          if (map.get(sessionId) !== deferred) return map
          const next = new Map(map)
          next.delete(sessionId)
          return next
        })

      const awaitStartedEntry = Effect.fn("acn.agent-runtime.await-started-entry")(function* (
        sessionId: string,
        deferred: RuntimeStartDeferred,
      ) {
        const entry = yield* Deferred.await(deferred)
        return (yield* touchEntryNow(sessionId)) ?? entry
      })

      const startEntry = Effect.fn("acn.agent-runtime.start-entry")(function* (request: RuntimeStartRequest) {
        const providedScope = Option.getOrUndefined(request.scope)
        const scope = providedScope ?? (yield* Scope.make())
        return yield* Effect.gen(function* () {
          const requestedCwd = yield* store.validateCwd(request.cwd)
          yield* runtimeOptions.write(request.sessionId, request.options)
          const session = yield* factory.createSession({
            sessionId: request.sessionId,
            cwd: requestedCwd,
            scope,
            options: request.options,
            visibility: request.visibility,
          })
          const now = Date.now()
          const storedMeta = yield* store.readMeta(request.sessionId)
          const createdAt = storedMeta ? Date.parse(storedMeta.created) || now : now
          const scratchpadPath = yield* store.getScratchpadPath(request.sessionId)
          const entry: RuntimeEntry = {
            id: request.sessionId,
            createdAt,
            updatedAt: now,
            title: storedMeta?.chatName ?? DEFAULT_CHAT_NAME,
            cwd: requestedCwd,
            scratchpadPath,
            session,
            scope,
          }
          yield* Ref.update(entries, (map) => new Map(map).set(request.sessionId, entry))
          yield* publishChange
          return entry
        }).pipe(
          Effect.onExit((exit) =>
            Exit.isSuccess(exit) || Option.isSome(request.scope)
              ? Effect.void
              : Scope.close(scope, Exit.void)
          ),
        )
      })

      const completeStart = (
        sessionId: string,
        deferred: RuntimeStartDeferred,
      ) =>
        Effect.onExit((exit: Exit.Exit<RuntimeEntry, SessionError>) =>
          clearStart(sessionId, deferred).pipe(
            Effect.zipRight(Deferred.done(deferred, exit))
          )
        )

      const getOrStart = Effect.fn("acn.agent-runtime.get-or-start")(function* (request: RuntimeStartRequest) {
        const existing = yield* getLive(request.sessionId)
        if (existing) {
          const touched = yield* touchEntryNow(request.sessionId)
          return touched ?? existing
        }

        const claim = yield* claimStart(request.sessionId)
        if (claim._tag === "joiner") {
          return yield* awaitStartedEntry(request.sessionId, claim.deferred)
        }

        return yield* startEntry(request).pipe(
          completeStart(request.sessionId, claim.deferred),
        )
      })

      const requireOrStart = Effect.fn("acn.agent-runtime.require-or-start")(function* (sessionId: string) {
        const existing = yield* getLive(sessionId)
        if (existing) {
          return (yield* touchEntryNow(sessionId)) ?? existing
        }

        const meta = yield* store.readMeta(sessionId)
        if (!meta) return yield* new SessionNotFound({ sessionId })

        const stored = yield* runtimeOptions.read(sessionId)
        const options = stored ?? normalizeSessionRuntimeOptions()
        return yield* getOrStart({
          sessionId,
          cwd: meta.workingDirectory,
          options,
          visibility: meta.visibility,
          scope: Option.none(),
        })
      })

      const dispose = Effect.fn("acn.agent-runtime.dispose")(function* (sessionId: string) {
        const entry = yield* Ref.modify(entries, (map): readonly [RuntimeEntry | null, Map<string, RuntimeEntry>] => {
          const current = map.get(sessionId) ?? null
          if (!current) return [null, map]

          const next = new Map(map)
          next.delete(sessionId)
          return [current, next]
        })
        if (!entry) return
        yield* Ref.update(retains, (map) => {
          if (!map.has(sessionId)) return map
          const next = new Map(map)
          next.delete(sessionId)
          return next
        })
        yield* publishChange
        yield* Scope.close(entry.scope, Exit.void)
      })

      const evictIdleSessions = Effect.fn("acn.agent-runtime.evict-idle")(function* () {
        const liveEntries = [...(yield* Ref.get(entries)).values()]
        const retainSnapshot = yield* Ref.get(retains)
        for (const entry of liveEntries) {
          if ((retainSnapshot.get(entry.id) ?? 0) > 0) continue

          const rootTurn = yield* entry.session.state.turn.getFork(null)
          if (rootTurn._tag === "active" || rootTurn._tag === "interrupting") {
            yield* touchEntry(entry.id)
            continue
          }

          const agentStatus = yield* entry.session.state.agentStatus.get()
          if ([...agentStatus.agents.values()].some((a) => a.status === "working")) {
            yield* touchEntry(entry.id)
            continue
          }

          const idleMs = Date.now() - entry.updatedAt
          if (idleMs >= SESSION_EVICT_GRACE_MS) {
            yield* dispose(entry.id)
            yield* Effect.logInfo("Evicted idle session", { sessionId: entry.id, idleMs })
          }
        }
      })

      const disposeAll = Effect.fn("acn.agent-runtime.dispose-all")(function* () {
        const ids = [...(yield* Ref.get(entries)).keys()]
        yield* Effect.forEach(ids, dispose, { discard: true })
      })
      yield* Effect.addFinalizer(() => disposeAll())

      return {
        getLive,
        getAllEntries: Effect.fn("acn.agent-runtime.get-all-entries")(function* () {
          return [...(yield* Ref.get(entries)).values()]
        }),
        getOrStart,
        requireOrStart,
        dispose,
        touchEntry,
        retainEntry,
        releaseEntry,
        evictIdleSessions,
        disposeAll,
        hasActiveWork: Effect.gen(function* () {
          const liveEntries = [...(yield* Ref.get(entries)).values()]
          for (const entry of liveEntries) {
            const rootTurn = yield* entry.session.state.turn.getFork(null)
            if (rootTurn._tag === "active" || rootTurn._tag === "interrupting") {
              return true
            }

            const agentStatus = yield* entry.session.state.agentStatus.get()
            if ([...agentStatus.agents.values()].some((agent) => agent.status === "working")) {
              return true
            }
          }
          return false
        }),
        count: Ref.get(entries).pipe(Effect.map((map) => map.size)),
        changes: Stream.fromPubSub(changes).pipe(Stream.map(() => undefined)),
      }
    }),
  )
