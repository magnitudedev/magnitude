import { Context, Cause, Effect, Layer, Option } from "effect"
import { resolve } from "path"
import { stat } from "fs/promises"
import { DEFAULT_CHAT_NAME } from "@magnitudedev/agent"
import { MagnitudeStorage, type StoredSessionMeta } from "@magnitudedev/storage"
import {
  InvalidSessionPath,
  SessionNotFound,
  SessionOperationFailed,
  type ListSessionsResult,
  type SessionError,
  type SessionCwdSummary,
  type SessionMetadata as ProtocolSessionMetadata,
} from "@magnitudedev/protocol"
import type { SessionExecutionContext } from "./session-types"

export interface SessionStoreApi {
  readonly createId: Effect.Effect<string>
  readonly readMeta: (sessionId: string) => Effect.Effect<StoredSessionMeta | null, SessionError>
  readonly readProtocolMeta: (sessionId: string) => Effect.Effect<ProtocolSessionMetadata | null, SessionError>
  readonly promoteDraft: (sessionId: string) => Effect.Effect<StoredSessionMeta, SessionError>
  readonly listDraftSessionIds: () => Effect.Effect<ReadonlyArray<string>, SessionError>
  readonly listProtocolMetas: (
    options?: { readonly cwd?: string; readonly query?: string; readonly cursor?: string; readonly limit?: number }
  ) => Effect.Effect<ListSessionsResult, SessionError>
  readonly listSessionCwds: () => Effect.Effect<ReadonlyArray<SessionCwdSummary>, SessionError>
  readonly deleteSessionFiles: (sessionId: string) => Effect.Effect<void, SessionError>
  readonly validateCwd: (cwd: string) => Effect.Effect<string, SessionError>
  readonly getScratchpadPath: (sessionId: string) => Effect.Effect<string, SessionError>
  readonly getExecutionContext: (sessionId: string) => Effect.Effect<SessionExecutionContext, SessionError>

}

export class SessionStore extends Context.Tag("SessionStore")<
  SessionStore,
  SessionStoreApi
>() {}

function storedMetaToProtocol(meta: StoredSessionMeta): ProtocolSessionMetadata {
  const createdAt = Date.parse(meta.created)
  const updatedAt = Date.parse(meta.updated)
  return {
    sessionId: meta.sessionId,
    title: meta.chatName,
    cwd: meta.workingDirectory,
    createdAt: Number.isFinite(createdAt) ? createdAt : Date.now(),
    updatedAt: Number.isFinite(updatedAt) ? updatedAt : Date.now(),
    messageCount: meta.messageCount ?? 0,
    lastMessage: meta.lastMessage ?? null,
  }
}

interface SessionCursor {
  readonly updatedAt: number
  readonly sessionId: string
}

const compareProtocolMetasNewestFirst = (
  left: ProtocolSessionMetadata,
  right: ProtocolSessionMetadata,
): number => {
  const updatedDelta = right.updatedAt - left.updatedAt
  if (updatedDelta !== 0) return updatedDelta
  return right.sessionId.localeCompare(left.sessionId)
}

const compareMetaToCursor = (
  meta: ProtocolSessionMetadata,
  cursor: SessionCursor,
): number => {
  const updatedDelta = cursor.updatedAt - meta.updatedAt
  if (updatedDelta !== 0) return updatedDelta
  return cursor.sessionId.localeCompare(meta.sessionId)
}

const encodeSessionCursor = (meta: ProtocolSessionMetadata): string =>
  `${meta.updatedAt}:${encodeURIComponent(meta.sessionId)}`

const decodeSessionCursor = (cursor: string): SessionCursor | null => {
  const separatorIndex = cursor.indexOf(":")
  if (separatorIndex <= 0) return null
  const updatedAt = Number(cursor.slice(0, separatorIndex))
  if (!Number.isFinite(updatedAt)) return null
  return {
    updatedAt,
    sessionId: decodeURIComponent(cursor.slice(separatorIndex + 1)),
  }
}

const clampSessionPageLimit = (limit: number | undefined): number =>
  Math.min(100, Math.max(1, Math.trunc(limit ?? 50)))

const toSessionErrorFromPlatform = (operation: string) => (cause: unknown): SessionError =>
  new SessionOperationFailed({
    operation,
    reason: Cause.pretty(Cause.fail(cause)),
  })

export const defaultStoredMeta = (
  sessionId: string,
  workingDirectory: string,
  version: string,
  now: string,
  visibility: StoredSessionMeta["visibility"] = "visible",
): StoredSessionMeta => ({
  sessionId,
  chatName: DEFAULT_CHAT_NAME,
  workingDirectory,
  visibility,
  gitBranch: null,
  created: now,
  updated: now,
  initialVersion: version,
  lastActiveVersion: version,
  firstUserMessage: null,
  lastMessage: null,
  messageCount: 0,
})

export const SessionStoreLive = Layer.effect(
  SessionStore,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage

    const readMeta = Effect.fn("acn.session-store.read-meta")(function* (sessionId: string) {
      return yield* storage.sessions.readMeta(sessionId).pipe(
        Effect.mapError(toSessionErrorFromPlatform(`read meta ${sessionId}`))
      )
    })

    const readProtocolMeta = Effect.fn("acn.session-store.read-protocol-meta")(function* (sessionId: string) {
      const meta = yield* readMeta(sessionId)
      return meta ? storedMetaToProtocol(meta) : null
    })

    const readAllProtocolMetas = Effect.fn("acn.session-store.read-all-protocol-metas")(function* () {
      const ids = yield* storage.sessions.listSessionIds().pipe(
        Effect.mapError(toSessionErrorFromPlatform("list sessions")),
      )
      const metas: ProtocolSessionMetadata[] = []
      for (const id of ids) {
        const rawMeta = yield* readMeta(id)
        if (!rawMeta || rawMeta.visibility !== "visible") continue
        metas.push(storedMetaToProtocol(rawMeta))
      }
      return metas
    })

    return {
      createId: Effect.sync(() => storage.sessions.createTimestampSessionId()),

      readMeta,

      readProtocolMeta,

      promoteDraft: Effect.fn("acn.session-store.promote-draft")(function* (sessionId: string) {
        const existing = yield* readMeta(sessionId)
        if (!existing) return yield* new SessionNotFound({ sessionId })
        return yield* storage.sessions.updateMeta(sessionId, (current) => {
          const now = new Date().toISOString()
          return {
            ...(current ?? existing),
            visibility: "visible",
            updated: now,
          }
        }).pipe(
          Effect.mapError(toSessionErrorFromPlatform(`promote draft ${sessionId}`))
        )
      }),

      listDraftSessionIds: Effect.fn("acn.session-store.list-draft-session-ids")(function* () {
        const ids = yield* storage.sessions.listSessionIds().pipe(
          Effect.mapError(toSessionErrorFromPlatform("list draft sessions")),
        )
        const draftIds: string[] = []
        for (const id of ids) {
          const meta = yield* readMeta(id)
          if (meta?.visibility === "draft") draftIds.push(id)
        }
        return draftIds
      }),

      listProtocolMetas: Effect.fn("acn.session-store.list-protocol-metas")(function* (options) {
        const cwd = options?.cwd ? resolve(options.cwd) : null
        const query = options?.query?.trim().toLowerCase() ?? ""
        const cursor = options?.cursor ? decodeSessionCursor(options.cursor) : null
        const limit = clampSessionPageLimit(options?.limit)
        const metas = yield* readAllProtocolMetas()
        const filtered = metas
          .filter((meta) => !cwd || resolve(meta.cwd) === cwd)
          .filter((meta) =>
            !query ||
            (meta.title ?? "").toLowerCase().includes(query) ||
            meta.cwd.toLowerCase().includes(query)
          )
          .sort(compareProtocolMetasNewestFirst)
        const afterCursor = cursor
          ? filtered.filter((meta) => compareMetaToCursor(meta, cursor) > 0)
          : filtered
        const pageWindow = afterCursor.slice(0, limit + 1)
        const items = pageWindow.slice(0, limit)
        const hasMore = pageWindow.length > limit
        const last = items[items.length - 1]
        return {
          items,
          nextCursor: hasMore && last
            ? Option.some(encodeSessionCursor(last))
            : Option.none(),
          hasMore,
        }
      }),

      listSessionCwds: Effect.fn("acn.session-store.list-session-cwds")(function* () {
        const metas = yield* readAllProtocolMetas()
        const byCwd = new Map<string, { updatedAt: number; sessionCount: number }>()
        for (const meta of metas) {
          const cwd = resolve(meta.cwd)
          const existing = byCwd.get(cwd)
          byCwd.set(cwd, {
            updatedAt: Math.max(existing?.updatedAt ?? 0, meta.updatedAt),
            sessionCount: (existing?.sessionCount ?? 0) + 1,
          })
        }
        return [...byCwd.entries()]
          .map(([cwd, summary]) => ({ cwd, ...summary }))
          .sort((left, right) => {
            const updatedDelta = right.updatedAt - left.updatedAt
            if (updatedDelta !== 0) return updatedDelta
            return left.cwd.localeCompare(right.cwd)
          })
      }),

      deleteSessionFiles: Effect.fn("acn.session-store.delete-session-files")(function* (sessionId) {
        yield* storage.sessions.deleteSession(sessionId).pipe(
          Effect.mapError(toSessionErrorFromPlatform(`delete session ${sessionId}`))
        )
      }),

      validateCwd: Effect.fn("acn.session-store.validate-cwd")(function* (cwd) {
        const requestedCwd = resolve(cwd)
        const cwdStat = yield* Effect.tryPromise({
          try: () => stat(requestedCwd),
          catch: () => new SessionOperationFailed({
            operation: `stat ${requestedCwd}`,
            reason: "path not found",
          }),
        })
        if (!cwdStat.isDirectory()) {
          return yield* new InvalidSessionPath({ path: requestedCwd })
        }
        return requestedCwd
      }),

      getScratchpadPath: (sessionId) =>
        Effect.sync(() => storage.sessions.paths.sessionScratchpad(sessionId)),

      getExecutionContext: Effect.fn("acn.session-store.get-execution-context")(function* (sessionId) {
        const meta = yield* readMeta(sessionId)
        if (!meta) return yield* new SessionNotFound({ sessionId })
        const cwd = meta.workingDirectory
        return {
          cwd,
          projectRoot: cwd,
          scratchpadPath: storage.sessions.paths.sessionScratchpad(sessionId),
        }
      }),

    }
  })
)
