import { Context, Effect, Layer, Option, Ref } from "effect"
import { resolve } from "node:path"
import {
  SessionAlreadyExists,
  SessionNotFound,
  SessionOperationFailed,
  type SessionError,
  type SessionMetadata as ProtocolSessionMetadata,
  type SessionOptions,
} from "@magnitudedev/protocol"
import { AgentRuntime, type RuntimeStartRequest } from "./agent-runtime"
import { sessionErrorMessage } from "./session-errors"
import { SessionRuntimeOptionsStore, type SessionRuntimeOptions } from "./session-runtime-options"
import { SessionStore } from "./session-store"

const READY_DRAFT_TTL_MS = 10 * 60 * 1000
const SWEEP_INTERVAL = "60 seconds"

type DraftPhase = "preloading" | "ready" | "claiming"

interface DraftEntry {
  readonly key: string
  readonly cwd: string
  readonly options: SessionRuntimeOptions
  readonly ownerId: string | null
  readonly sessionId: string
  readonly createdAt: number
  readonly touchedAt: number
  readonly phase: DraftPhase
}

export interface DraftClaim {
  readonly key: string
  readonly sessionId: string
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

const makeDraftKey = (input: {
  readonly cwd: string
  readonly options: SessionRuntimeOptions
  readonly ownerId: string | null
}): string => JSON.stringify(input)

const staleDraftError = (sessionId: string, reason: string) =>
  new SessionOperationFailed({
    operation: `draft session ${sessionId}`,
    reason,
  })

export const SessionDraftsLive: Layer.Layer<
  SessionDrafts,
  never,
  AgentRuntime | SessionStore | SessionRuntimeOptionsStore
> = Layer.scoped(
  SessionDrafts,
  Effect.gen(function* () {
    const runtime = yield* AgentRuntime
    const store = yield* SessionStore
    const runtimeOptions = yield* SessionRuntimeOptionsStore
    const entries = yield* Ref.make(new Map<string, DraftEntry>())

    const deriveKey = (input: {
      readonly cwd: string
      readonly options?: SessionOptions
      readonly ownerId?: string | null
    }) =>
      store.validateCwd(resolve(input.cwd)).pipe(
        Effect.map((cwd) => {
          const options = runtimeOptions.normalize(input.options)
          const ownerId = input.ownerId ?? null
          return {
            key: makeDraftKey({ cwd, options, ownerId }),
            cwd,
            options,
            ownerId,
          }
        }),
      )

    const removeExact = (entry: DraftEntry, allowed: ReadonlySet<DraftPhase>) =>
      Ref.modify(entries, (current) => {
        const found = current.get(entry.key)
        if (!found || found.sessionId !== entry.sessionId || !allowed.has(found.phase)) {
          return [Option.none<DraftEntry>(), current] as const
        }
        const next = new Map(current)
        next.delete(entry.key)
        return [Option.some(found), next] as const
      })

    const cleanupEmptyDraft = Effect.fn("acn.session-drafts.cleanup-empty")(function* (
      entry: DraftEntry,
    ) {
      const meta = yield* store.readMeta(entry.sessionId)
      if (meta && (meta.messageCount ?? 0) > 0) return false
      yield* runtime.deleteSession(entry.sessionId, store.deleteSessionFiles(entry.sessionId))
      return true
    })

    const abandonOtherOwnerEntries = (ownerId: string | null, keepKey: string) =>
      ownerId === null
        ? Effect.void
        : Effect.gen(function* () {
            for (const candidate of (yield* Ref.get(entries)).values()) {
              if (
                candidate.ownerId !== ownerId ||
                candidate.key === keepKey ||
                candidate.phase === "claiming"
              ) {
                continue
              }
              const removed = yield* removeExact(
                candidate,
                new Set<DraftPhase>(["preloading", "ready"]),
              )
              if (Option.isSome(removed)) yield* cleanupEmptyDraft(removed.value)
            }
          })

    const ensureRecord = Effect.fn("acn.session-drafts.ensure-record")(function* (input: {
      readonly cwd: string
      readonly sessionId?: string
      readonly options?: SessionOptions
      readonly ownerId?: string | null
    }) {
      const derived = yield* deriveKey(input)
      const now = Date.now()
      const candidate: DraftEntry = {
        ...derived,
        sessionId: input.sessionId ?? (yield* store.createId),
        createdAt: now,
        touchedAt: now,
        phase: "preloading",
      }
      const entry = yield* Ref.modify(entries, (current) => {
        const existing = current.get(derived.key)
        if (existing) {
          const touched = { ...existing, touchedAt: now }
          return [touched, new Map(current).set(derived.key, touched)] as const
        }
        return [candidate, new Map(current).set(derived.key, candidate)] as const
      })
      yield* abandonOtherOwnerEntries(derived.ownerId, derived.key)
      return entry
    })

    const startRequest = (entry: DraftEntry): RuntimeStartRequest => ({
      sessionId: entry.sessionId,
      cwd: entry.cwd,
      options: entry.options,
      visibility: "draft",
    })

    const initialize = Effect.fn("acn.session-drafts.initialize")(function* (entry: DraftEntry) {
      const meta = yield* store.readMeta(entry.sessionId)
      if (meta && meta.visibility !== "draft") {
        return yield* new SessionAlreadyExists({ sessionId: entry.sessionId })
      }
      yield* runtime.withSessionRequest(startRequest(entry), "draft-initialize", () => Effect.void)
    })

    const markReady = (entry: DraftEntry) =>
      Ref.modify(entries, (current) => {
        const found = current.get(entry.key)
        if (!found || found.sessionId !== entry.sessionId) return [false, current] as const
        if (found.phase === "claiming") return [true, current] as const
        const ready: DraftEntry = { ...found, phase: "ready", touchedAt: Date.now() }
        return [true, new Map(current).set(entry.key, ready)] as const
      })

    const restoreClaim = (claim: DraftClaim) =>
      Ref.update(entries, (current) => {
        const found = current.get(claim.key)
        if (!found || found.sessionId !== claim.sessionId || found.phase !== "claiming") {
          return current
        }
        return new Map(current).set(claim.key, {
          ...found,
          phase: "ready",
          touchedAt: Date.now(),
        })
      })

    const logLifecycleError = (message: string) => (error: SessionError) =>
      Effect.logWarning(message).pipe(
        Effect.annotateLogs({ error: sessionErrorMessage(error) }),
      )

    const preload = Effect.fn("acn.session-drafts.preload")(function* (input: {
      readonly cwd: string
      readonly options?: SessionOptions
      readonly ownerId?: string | null
    }) {
      const entry = yield* ensureRecord(input)
      const result = yield* Effect.either(initialize(entry))
      if (result._tag === "Left") {
        const removed = yield* removeExact(entry, new Set<DraftPhase>(["preloading", "ready"]))
        if (Option.isSome(removed)) yield* cleanupEmptyDraft(removed.value)
        return yield* result.left
      }
      if (!(yield* markReady(entry))) {
        return yield* staleDraftError(entry.sessionId, "draft was released during preload")
      }
      return { sessionId: entry.sessionId }
    })

    const release = Effect.fn("acn.session-drafts.release")(function* (input: {
      readonly cwd: string
      readonly options?: SessionOptions
      readonly ownerId?: string | null
    }) {
      const { key } = yield* deriveKey(input)
      const current = (yield* Ref.get(entries)).get(key)
      if (!current) return
      const removed = yield* removeExact(current, new Set<DraftPhase>(["preloading", "ready"]))
      if (Option.isNone(removed)) return
      if (!(yield* cleanupEmptyDraft(removed.value))) {
        yield* Ref.update(entries, (all) =>
          all.has(key) ? all : new Map(all).set(key, removed.value),
        )
      }
    })

    const claim = Effect.fn("acn.session-drafts.claim")(function* (input: {
      readonly cwd: string
      readonly sessionId?: string
      readonly options?: SessionOptions
      readonly ownerId?: string | null
    }) {
      const entry = yield* ensureRecord(input)
      const claimed = yield* Ref.modify(entries, (current) => {
        const found = current.get(entry.key)
        if (!found || found.sessionId !== entry.sessionId || found.phase === "claiming") {
          return [false, current] as const
        }
        return [
          true,
          new Map(current).set(entry.key, {
            ...found,
            phase: "claiming",
            touchedAt: Date.now(),
          }),
        ] as const
      })
      if (!claimed) return yield* new SessionAlreadyExists({ sessionId: entry.sessionId })

      const result = yield* Effect.either(initialize(entry))
      if (result._tag === "Left") {
        yield* restoreClaim({ key: entry.key, sessionId: entry.sessionId })
        return yield* result.left
      }
      return { key: entry.key, sessionId: entry.sessionId }
    })

    const promote = Effect.fn("acn.session-drafts.promote")(function* (claim: DraftClaim) {
      yield* store.promoteDraft(claim.sessionId)
      const current = (yield* Ref.get(entries)).get(claim.key)
      if (current?.sessionId === claim.sessionId && current.phase === "claiming") {
        yield* Ref.update(entries, (all) => {
          const next = new Map(all)
          next.delete(claim.key)
          return next
        })
      }
      const stored = yield* store.readProtocolMeta(claim.sessionId)
      if (!stored) return yield* new SessionNotFound({ sessionId: claim.sessionId })
      return stored
    })

    const sweepExpiredDrafts = Effect.fn("acn.session-drafts.sweep")(function* () {
      const now = Date.now()
      for (const entry of (yield* Ref.get(entries)).values()) {
        if (entry.phase !== "ready" || now - entry.touchedAt < READY_DRAFT_TTL_MS) continue
        const removed = yield* removeExact(entry, new Set<DraftPhase>(["ready"]))
        if (Option.isSome(removed) && !(yield* cleanupEmptyDraft(removed.value))) {
          yield* Ref.update(entries, (all) =>
            all.has(entry.key) ? all : new Map(all).set(entry.key, removed.value),
          )
        }
      }
    })

    yield* Effect.forever(
      Effect.sleep(SWEEP_INTERVAL).pipe(
        Effect.zipRight(sweepExpiredDrafts()),
        Effect.catchAll(logLifecycleError("Failed to sweep expired draft sessions")),
      ),
    ).pipe(Effect.forkScoped)

    return {
      preload,
      release,
      claim,
      promote,
      releaseClaim: restoreClaim,
    } satisfies SessionDraftsApi
  }),
)
