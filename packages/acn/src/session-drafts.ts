import { Context, Data, Deferred, Effect, Layer, Ref, Scope, Exit, ExecutionStrategy, Option } from "effect"
import type { CloseableScope } from "effect/Scope"
import { resolve } from "path"
import {
  SessionAlreadyExists,
  SessionNotFound,
  SessionOperationFailed,
  type SessionError,
  type SessionMetadata as ProtocolSessionMetadata,
  type SessionOptions,
} from "@magnitudedev/protocol"
import { FSM } from "@magnitudedev/utils"
import type { RuntimeEntry } from "./session-types"
import { AgentRuntime } from "./agent-runtime"
import { SessionStore } from "./session-store"
import { sessionErrorMessage } from "./session-errors"
import {
  SessionRuntimeOptionsStore,
  type SessionRuntimeOptions,
} from "./session-runtime-options"

const { defineFSM } = FSM

const READY_DRAFT_TTL_MS = 10 * 60 * 1000
const SWEEP_INTERVAL = "60 seconds"

interface DraftKeyParts {
  readonly cwd: string
  readonly options: SessionRuntimeOptions
  readonly ownerId: string | null
}

/** Fields present on every draft state. The FSM state classes ARE the entry. */
export class DraftPreloading extends Data.TaggedClass("preloading")<{
  readonly key: string
  readonly cwd: string
  readonly options: SessionRuntimeOptions
  readonly ownerId: string | null
  readonly sessionId: string
  readonly scope: CloseableScope
  readonly deferred: Deferred.Deferred<RuntimeEntry, SessionError>
  readonly createdAt: number
  readonly touchedAt: number
}> {}

export class DraftReady extends Data.TaggedClass("ready")<{
  readonly key: string
  readonly cwd: string
  readonly options: SessionRuntimeOptions
  readonly ownerId: string | null
  readonly sessionId: string
  readonly scope: CloseableScope
  readonly deferred: Deferred.Deferred<RuntimeEntry, SessionError>
  readonly createdAt: number
  readonly touchedAt: number
}> {}

export class DraftClaiming extends Data.TaggedClass("claiming")<{
  readonly key: string
  readonly cwd: string
  readonly options: SessionRuntimeOptions
  readonly ownerId: string | null
  readonly sessionId: string
  readonly scope: CloseableScope
  readonly deferred: Deferred.Deferred<RuntimeEntry, SessionError>
  readonly createdAt: number
  readonly touchedAt: number
}> {}

export const DraftLifecycle = defineFSM(
  {
    preloading: DraftPreloading,
    ready: DraftReady,
    claiming: DraftClaiming,
  },
  {
    preloading: ["ready", "claiming"],
    ready: ["claiming"],
    claiming: ["ready"],
  } as const,
)

export type DraftState = DraftPreloading | DraftReady | DraftClaiming

export interface DraftClaim {
  readonly key: string
  readonly sessionId: string
  readonly entry: RuntimeEntry
}

export interface SessionDraftsApi {
  readonly preload: (input: {
    readonly cwd: string
    readonly options?: SessionOptions
    readonly ownerId?: string | null
  }) => Effect.Effect<{ readonly sessionId: string }, SessionError>
  readonly release: (input: {
    readonly cwd: string
    readonly options?: SessionOptions
    readonly ownerId?: string | null
  }) => Effect.Effect<void, SessionError>
  readonly claim: (input: {
    readonly cwd: string
    readonly sessionId?: string
    readonly options?: SessionOptions
    readonly ownerId?: string | null
  }) => Effect.Effect<DraftClaim, SessionError>
  readonly promote: (claim: DraftClaim) => Effect.Effect<ProtocolSessionMetadata, SessionError>
  readonly releaseClaim: (claim: DraftClaim) => Effect.Effect<void, SessionError>
}

export class SessionDrafts extends Context.Tag("SessionDrafts")<
  SessionDrafts,
  SessionDraftsApi
>() {}

const makeDraftKey = (parts: DraftKeyParts): string =>
  JSON.stringify(parts)

const staleDraftError = (sessionId: string, reason: string) =>
  new SessionOperationFailed({
    operation: `draft session ${sessionId}`,
    reason,
  })

export const SessionDraftsLive: Layer.Layer<
  SessionDrafts,
  never,
  AgentRuntime | SessionStore | SessionRuntimeOptionsStore
> =
  Layer.scoped(
    SessionDrafts,
    Effect.gen(function* () {
      const managerScope = yield* Effect.scopeWith((scope) => Effect.succeed(scope))
      const runtime = yield* AgentRuntime
      const store = yield* SessionStore
      const runtimeOptions = yield* SessionRuntimeOptionsStore
      const entries = yield* Ref.make(new Map<string, DraftState>())

      // --- Helpers ---

      const deriveDraftKey = (input: {
        readonly cwd: string
        readonly options?: SessionOptions
        readonly ownerId?: string | null
      }): Effect.Effect<{ readonly key: string; readonly cwd: string; readonly options: SessionRuntimeOptions }, SessionError> =>
        Effect.map(store.validateCwd(resolve(input.cwd)), (cwd) => {
          const options = runtimeOptions.normalize(input.options)
          return {
            key: makeDraftKey({
              cwd,
              options,
              ownerId: input.ownerId ?? null,
            }),
            cwd,
            options,
          }
        })

      const removePreloadingEntry = (entry: DraftState): Effect.Effect<Option.Option<DraftState>> =>
        Ref.modify(entries, (map): readonly [Option.Option<DraftState>, Map<string, DraftState>] => {
          const current = map.get(entry.key)
          if (!current || current.sessionId !== entry.sessionId || current._tag !== "preloading") {
            return [Option.none(), map]
          }

          const next = new Map(map)
          next.delete(entry.key)
          return [Option.some(current), next]
        })

      const removeStartedEntry = (entry: DraftState): Effect.Effect<Option.Option<DraftState>> =>
        Ref.modify(entries, (map): readonly [Option.Option<DraftState>, Map<string, DraftState>] => {
          const current = map.get(entry.key)
          if (!current || current.sessionId !== entry.sessionId) {
            return [Option.none(), map]
          }
          if (current._tag !== "preloading" && current._tag !== "claiming") {
            return [Option.none(), map]
          }

          const next = new Map(map)
          next.delete(entry.key)
          return [Option.some(current), next]
        })

      const removeUnclaimedOwnerEntry = (entry: DraftState): Effect.Effect<Option.Option<DraftState>> =>
        Ref.modify(entries, (map): readonly [Option.Option<DraftState>, Map<string, DraftState>] => {
          const current = map.get(entry.key)
          if (!current || current.sessionId !== entry.sessionId) {
            return [Option.none(), map]
          }
          if (current._tag !== "preloading" && current._tag !== "ready") {
            return [Option.none(), map]
          }

          const next = new Map(map)
          next.delete(entry.key)
          return [Option.some(current), next]
        })

      const removeNonClaimingEntry = (key: string): Effect.Effect<Option.Option<DraftState>> =>
        Ref.modify(entries, (map): readonly [Option.Option<DraftState>, Map<string, DraftState>] => {
          const current = map.get(key)
          if (!current || current._tag === "claiming") {
            return [Option.none(), map]
          }

          const next = new Map(map)
          next.delete(key)
          return [Option.some(current), next]
        })

      const removeClaimingEntry = (claim: DraftClaim): Effect.Effect<Option.Option<DraftState>> =>
        Ref.modify(entries, (map): readonly [Option.Option<DraftState>, Map<string, DraftState>] => {
          const current = map.get(claim.key)
          if (!current || current.sessionId !== claim.sessionId || current._tag !== "claiming") {
            return [Option.none(), map]
          }

          const next = new Map(map)
          next.delete(claim.key)
          return [Option.some(current), next]
        })

      const removeReadyEntry = (entry: DraftState): Effect.Effect<Option.Option<DraftState>> =>
        Ref.modify(entries, (map): readonly [Option.Option<DraftState>, Map<string, DraftState>] => {
          const current = map.get(entry.key)
          if (!current || current.sessionId !== entry.sessionId || current._tag !== "ready") {
            return [Option.none(), map]
          }

          const next = new Map(map)
          next.delete(entry.key)
          return [Option.some(current), next]
        })

      const markEntryReady = (entry: DraftState): Effect.Effect<void> =>
        Ref.update(entries, (map) => {
          const current = map.get(entry.key)
          if (!current || current.sessionId !== entry.sessionId) return map
          if (current._tag === "claiming") return map
          if (current._tag !== "preloading") return map

          const next = DraftLifecycle.transition(current, "ready", { touchedAt: Date.now() })
          return new Map(map).set(entry.key, next)
        })

      const restoreClaimToReady = (claim: DraftClaim): Effect.Effect<void> =>
        Ref.update(entries, (map) => {
          const current = map.get(claim.key)
          if (!current || current.sessionId !== claim.sessionId || current._tag !== "claiming") {
            return map
          }
          const next = DraftLifecycle.transition(current, "ready", { touchedAt: Date.now() })
          return new Map(map).set(claim.key, next)
        })

      const reinsertIfAbsent = (entry: DraftState): Effect.Effect<void> =>
        Ref.update(entries, (map) => {
          if (map.has(entry.key)) return map
          return new Map(map).set(entry.key, entry)
        })

      const logDraftLifecycleError = (
        message: string,
        fields: { readonly sessionId?: string; readonly key?: string; readonly phase?: string },
      ) =>
        (error: SessionError): Effect.Effect<void> =>
          Effect.logWarning(message).pipe(
            Effect.annotateLogs({
              ...fields,
              error: sessionErrorMessage(error),
            }),
          )

      /**
       * Abandon a draft entry: fail the deferred, close the scope (disposes
       * the runtime automatically), and explicitly delete store files.
       * The entry has already been removed from the map by the caller's
       * atomic `Ref.modify` delete.
       */
      const abandonEntry = Effect.fn("acn.session-drafts.abandon-entry")(function* (
        entry: DraftState,
        reason: string,
      ) {
        yield* Deferred.fail(entry.deferred, staleDraftError(entry.sessionId, reason))
        yield* runtime.dispose(entry.sessionId)
        yield* Scope.close(entry.scope, Exit.void)
        yield* store.deleteSessionFiles(entry.sessionId)
      })

      const rejectBeforeStart = Effect.fn("acn.session-drafts.reject-before-start")(function* (
        entry: DraftState,
        error: SessionError,
      ) {
        yield* Deferred.fail(entry.deferred, error)
        yield* removePreloadingEntry(entry)
        yield* Scope.close(entry.scope, Exit.void)
      })

      const failStartedEntry = Effect.fn("acn.session-drafts.fail-started-entry")(function* (
        entry: DraftState,
        error: SessionError,
      ) {
        yield* Deferred.fail(entry.deferred, error)
        const removed = yield* removeStartedEntry(entry)
        if (Option.isSome(removed)) {
          yield* Scope.close(entry.scope, Exit.void)
          yield* store.deleteSessionFiles(entry.sessionId)
        }
      })

      const abandonOtherOwnerEntries = (ownerId: string | null, keepKey: string) =>
        ownerId === null
          ? Effect.void
          : Effect.gen(function* () {
            const snapshot = [...(yield* Ref.get(entries)).values()]
            for (const entry of snapshot) {
              if (entry.ownerId !== ownerId) continue
              if (entry.key === keepKey) continue
              // Skip claiming — someone owns it. The atomic delete below also
              // enforces this, but skipping avoids unnecessary work.
              if (entry._tag === "claiming") continue
              const removed = yield* removeUnclaimedOwnerEntry(entry)
              if (Option.isSome(removed)) {
                yield* abandonEntry(removed.value, "draft owner selected a different working directory or options")
              }
            }
          })

      const startEntry = (entry: DraftState) =>
        Effect.gen(function* () {
          const existingLive = yield* runtime.getLive(entry.sessionId)
          if (existingLive) {
            yield* rejectBeforeStart(entry, new SessionAlreadyExists({ sessionId: entry.sessionId }))
            return
          }

          const existingMeta = yield* store.readMeta(entry.sessionId)
          if (existingMeta) {
            yield* rejectBeforeStart(entry, new SessionAlreadyExists({ sessionId: entry.sessionId }))
            return
          }

          yield* runtime.getOrStart({
            sessionId: entry.sessionId,
            cwd: entry.cwd,
            options: entry.options,
            visibility: "draft",
            scope: Option.some(entry.scope),
          }).pipe(
            Effect.matchEffect({
              onFailure: (error) =>
                failStartedEntry(entry, error).pipe(
                  Effect.catchAll(logDraftLifecycleError(
                    "Failed to clean up draft after startup failure",
                    { sessionId: entry.sessionId, key: entry.key, phase: entry._tag },
                  )),
                ),
              onSuccess: (runtimeEntry) =>
                Effect.gen(function* () {
                  // Success: transition preloading → ready, BUT only if not claiming.
                  // If claim won the race (preloading → claiming while we were in
                  // getOrStart), this fails cleanly — claim keeps ownership. We still
                  // resolve the deferred so claim (suspended on awaitEntry) can resume.
                  // If the entry was already deleted (released/abandoned),
                  // Deferred.succeed is a no-op on the already-failed deferred. Safe
                  // either way.
                  yield* markEntryReady(entry)
                  yield* Deferred.succeed(entry.deferred, runtimeEntry)
                }),
            }),
          )
        }).pipe(
          Effect.catchAll((error) =>
            rejectBeforeStart(entry, error)
          ),
        )

      const ensureEntry = Effect.fn("acn.session-drafts.ensure-entry")(function* (input: {
        readonly cwd: string
        readonly sessionId?: string
        readonly options?: SessionOptions
        readonly ownerId?: string | null
      }) {
        const { key, cwd, options } = yield* deriveDraftKey(input)
        const ownerId = input.ownerId ?? null

        const deferred = yield* Deferred.make<RuntimeEntry, SessionError>()
        const sessionId = input.sessionId ?? (yield* store.createId)
        const now = Date.now()
        const draftScope = yield* Scope.fork(managerScope, ExecutionStrategy.sequential)

        const createdEntry: DraftPreloading = new DraftPreloading({
          key,
          cwd,
          options,
          ownerId,
          sessionId,
          scope: draftScope,
          deferred,
          createdAt: now,
          touchedAt: now,
        })

        // Single atomic insert-or-touch: if the key already exists, bump
        // touchedAt via an FSM hold; otherwise insert the new preloading entry.
        // If a key already exists, the new scope is unused — close it.
        const { entry, shouldStart } = yield* Ref.modify(entries, (map): readonly [{ readonly entry: DraftState; readonly shouldStart: boolean }, Map<string, DraftState>] => {
          const existing = map.get(key)
          if (existing) {
            const touched = DraftLifecycle.hold(existing, { touchedAt: Date.now() })
            const next = new Map(map).set(key, touched)
            return [{ entry: touched, shouldStart: false }, next]
          }
          const next = new Map(map).set(key, createdEntry)
          return [{ entry: createdEntry, shouldStart: true }, next]
        })

        // If we didn't use the scope (existing entry found), close it.
        if (!shouldStart) {
          yield* Scope.close(draftScope, Exit.void)
        }

        yield* abandonOtherOwnerEntries(ownerId, key)

        if (shouldStart) {
          // Fork startEntry into the draft scope — closing the draft scope
          // interrupts in-flight startup. No side-channel needed.
          yield* Effect.forkIn(startEntry(entry), entry.scope)
        }

        return entry
      })

      const awaitEntry = (entry: DraftState) =>
        Deferred.await(entry.deferred)

      const release = Effect.fn("acn.session-drafts.release")(function* (input: {
        readonly cwd: string
        readonly options?: SessionOptions
        readonly ownerId?: string | null
      }) {
        const { key } = yield* deriveDraftKey(input)
        // Can only release entries that are NOT claiming. The atomic delete
        // checks _tag in ["preloading", "ready"] synchronously — if claim
        // already transitioned to "claiming", this fails cleanly and no
        // abandonment happens.
        const removed = yield* removeNonClaimingEntry(key)
        if (Option.isSome(removed)) {
          const entry = removed.value
          // Guard: if the draft already has messages (e.g. promote failed
          // after sendUserMessage succeeded, then releaseClaim reverted it
          // to ready), do NOT destroy it. The sweep and startup cleanup will
          // handle it — sweeps skip messageCount > 0, startup promotes it.
          const meta = yield* store.readMeta(entry.sessionId)
          if (meta && (meta.messageCount ?? 0) > 0) {
            // Re-insert the entry so it stays tracked for sweeps/startup.
            yield* reinsertIfAbsent(entry)
            return
          }
          yield* abandonEntry(entry, "draft preload released")
        }
      })

      const claim = Effect.fn("acn.session-drafts.claim")(function* (input: {
        readonly cwd: string
        readonly sessionId?: string
        readonly options?: SessionOptions
        readonly ownerId?: string | null
      }) {
        const entry = yield* ensureEntry(input)

        // Mark claiming atomically BEFORE awaiting — this is the race fix.
        // Valid from both "preloading" and "ready" (both are in the transition
        // matrix). The session-id guard prevents a stale ensureEntry result
        // from transitioning a replacement entry that was inserted by a later
        // call after a concurrent release deleted the original.
        const claimed = yield* Ref.modify(entries, (map) => {
          const current = map.get(entry.key)
          if (!current || current.sessionId !== entry.sessionId)
            return [false, map] as const
          if (current._tag === "claiming")
            return [false, map] as const
          if (!DraftLifecycle.canTransition(current._tag, "claiming"))
            return [false, map] as const
          const next = DraftLifecycle.transition(current, "claiming", { touchedAt: Date.now() })
          return [true, new Map(map).set(entry.key, next)] as const
        })
        if (!claimed) {
          return yield* new SessionAlreadyExists({ sessionId: entry.sessionId })
        }

        const liveEntry = yield* awaitEntry(entry)
        return { key: entry.key, sessionId: entry.sessionId, entry: liveEntry }
      })

      const promote = Effect.fn("acn.session-drafts.promote")(function* (claim: DraftClaim) {
        yield* store.promoteDraft(claim.sessionId)
        // Only remove if still claiming — prevents double-promote.
        // Do NOT close the scope. The scope stays open, the runtime stays
        // alive. The entry is now a regular session in the runtime map.
        yield* removeClaimingEntry(claim)
        const stored = yield* store.readProtocolMeta(claim.sessionId)
        if (!stored) return yield* new SessionNotFound({ sessionId: claim.sessionId })
        return stored
      })

      const releaseClaim = Effect.fn("acn.session-drafts.release-claim")(function* (claim: DraftClaim) {
        // Revert claiming → ready so the draft can be reused or swept.
        // Pure FSM transition — no scope close, no store delete.
        // The draft stays alive: scope open, runtime running, store intact.
        // If the entry is gone (promoted/abandoned) or in a different state,
        // this is a harmless no-op.
        yield* restoreClaimToReady(claim)
      })

      const sweepExpiredDrafts = Effect.fn("acn.session-drafts.sweep-expired-drafts")(function* () {
        const now = Date.now()
        const snapshot = [...(yield* Ref.get(entries)).values()]
        for (const entry of snapshot) {
          // Skip preloading (in-flight startup) and claiming (owned by a claimer).
          if (entry._tag !== "ready") continue
          if (now - entry.touchedAt < READY_DRAFT_TTL_MS) continue
          // Preserve drafts containing messages. Promotion remains owned by
          // the normal claim/send flow; a TTL sweep must never delete user data.
          const meta = yield* store.readMeta(entry.sessionId)
          if (meta && (meta.messageCount ?? 0) > 0) continue
          const removed = yield* removeReadyEntry(entry)
          if (Option.isSome(removed)) {
            yield* abandonEntry(removed.value, "draft ttl expired")
          }
        }
      })

      // Retained for an explicit maintenance/migration path. Do not run this
      // during layer construction: it enumerates every historical session.
      const cleanupOrphanedDrafts = Effect.fn("acn.session-drafts.cleanup-orphaned-drafts")(function* () {
        const draftIds = yield* store.listDraftSessionIds()
        for (const sessionId of draftIds) {
          const meta = yield* store.readMeta(sessionId)
          if (!meta) {
            yield* store.deleteSessionFiles(sessionId)
            continue
          }
          if ((meta.messageCount ?? 0) > 0) {
            yield* store.promoteDraft(sessionId)
          } else {
            yield* store.deleteSessionFiles(sessionId)
          }
        }
      })

      // Intentionally disabled: a daemon must not scan all session metadata
      // before it can serve requests.
      // yield* cleanupOrphanedDrafts()

      yield* Effect.forkIn(
        Effect.forever(
          Effect.sleep(SWEEP_INTERVAL).pipe(
            Effect.andThen(sweepExpiredDrafts()),
            Effect.catchAll(logDraftLifecycleError("Failed to sweep expired draft sessions", {})),
          ),
        ),
        managerScope,
      )

      return {
        preload: Effect.fn("acn.session-drafts.preload")(function* (input) {
          const entry = yield* ensureEntry(input)
          const liveEntry = yield* awaitEntry(entry)
          return { sessionId: liveEntry.id }
        }),
        release,
        claim,
        promote,
        releaseClaim,
      }
    }),
  )
