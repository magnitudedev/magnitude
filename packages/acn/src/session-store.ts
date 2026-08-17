import { Context, Cause, Effect, Layer, Option, PubSub, Stream } from "effect"
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
  type ProjectId,
} from "@magnitudedev/acn-protocol"
import type { SessionExecutionContext } from "./session-types"
import { sessionErrorMessage } from "./session-errors"
import { ProjectRegistry } from "./project-registry"

export interface SessionStoreApi {
  readonly createId: Effect.Effect<string>
  readonly readMeta: (sessionId: string) => Effect.Effect<StoredSessionMeta | null, SessionError>
  readonly readProtocolMeta: (sessionId: string) => Effect.Effect<ProtocolSessionMetadata | null, SessionError>
  readonly promoteDraft: (sessionId: string) => Effect.Effect<StoredSessionMeta, SessionError>
  readonly listDraftSessionIds: () => Effect.Effect<ReadonlyArray<string>, SessionError>
  readonly listProtocolMetas: (
    options?: {
      readonly cwd?: string
      readonly projectId?: ProjectId
      readonly includeClosed?: boolean
      readonly query?: string
      readonly cursor?: string
      readonly limit?: number
    }
  ) => Effect.Effect<ListSessionsResult, SessionError>
  readonly listAllProtocolMetas: () => Effect.Effect<
    ReadonlyArray<ProtocolSessionMetadata>,
    SessionError
  >
  readonly listSessionCwds: () => Effect.Effect<ReadonlyArray<SessionCwdSummary>, SessionError>
  readonly deleteSessionFiles: (sessionId: string) => Effect.Effect<void, SessionError>
  readonly validateCwd: (cwd: string) => Effect.Effect<string, SessionError>
  readonly getScratchpadPath: (sessionId: string) => Effect.Effect<string, SessionError>
  readonly getExecutionContext: (sessionId: string) => Effect.Effect<SessionExecutionContext, SessionError>
  readonly ensureProjectForCwd: (cwd: string) => Effect.Effect<ProjectId, SessionError>
  readonly resolveProjectSource: (projectId: ProjectId) => Effect.Effect<string, SessionError>
  readonly setSidebarOpen: (
    sessionId: string,
    open: boolean,
  ) => Effect.Effect<ProtocolSessionMetadata, SessionError>
  readonly changes: Stream.Stream<void>
}

export class SessionStore extends Context.Tag("SessionStore")<
  SessionStore,
  SessionStoreApi
>() {}

function storedMetaToProtocol(
  meta: StoredSessionMeta,
  sourceDirectory: string,
): ProtocolSessionMetadata {
  const createdAt = Date.parse(meta.created)
  const updatedAt = Date.parse(meta.updated)
  return {
    sessionId: meta.sessionId,
    projectId: meta.projectId,
    title: meta.chatName,
    cwd: sourceDirectory,
    sidebarOpen: meta.sidebarOpen,
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
  projectId: ProjectId,
  version: string,
  now: string,
  visibility: StoredSessionMeta["visibility"] = "visible",
): StoredSessionMeta => ({
  sessionId,
  projectId,
  chatName: DEFAULT_CHAT_NAME,
  workingDirectory,
  visibility,
  sidebarOpen: true,
  gitBranch: null,
  created: now,
  updated: now,
  initialVersion: version,
  lastActiveVersion: version,
  firstUserMessage: null,
  lastMessage: null,
  messageCount: 0,
})

export const SessionStoreLive = Layer.scoped(
  SessionStore,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const projects = yield* ProjectRegistry
    const changes = yield* PubSub.sliding<void>(1)
    const publishChange = PubSub.publish(changes, undefined).pipe(Effect.asVoid)

    const readMeta = Effect.fn("acn.session-store.read-meta")(function* (sessionId: string) {
      return yield* storage.sessions.readMeta(sessionId).pipe(
        Effect.mapError(toSessionErrorFromPlatform(`read meta ${sessionId}`))
      )
    })

    const readProtocolMeta = Effect.fn("acn.session-store.read-protocol-meta")(function* (sessionId: string) {
      const meta = yield* readMeta(sessionId)
      if (!meta) return null
      const project = yield* projects.get(meta.projectId).pipe(
        Effect.mapError((error) => new SessionOperationFailed({
          operation: `resolve project for session ${sessionId}`,
          reason: String(error),
        })),
      )
      return storedMetaToProtocol(meta, project.sourceDirectory)
    })

    const readMetaForListing = Effect.fn("acn.session-store.read-meta-for-listing")(function* (sessionId: string) {
      return yield* readMeta(sessionId).pipe(
        Effect.catchAll((error) =>
          Effect.logWarning("Skipping unreadable session metadata").pipe(
            Effect.annotateLogs({ sessionId, error: sessionErrorMessage(error) }),
            Effect.as(null),
          )
        ),
      )
    })

    const readAllProtocolMetas = Effect.fn("acn.session-store.read-all-protocol-metas")(function* () {
      const ids = yield* storage.sessions.listSessionIds().pipe(
        Effect.mapError(toSessionErrorFromPlatform("list sessions")),
      )
      const metas: ProtocolSessionMetadata[] = []
      for (const id of ids) {
        const rawMeta = yield* readMetaForListing(id)
        if (!rawMeta || rawMeta.visibility !== "visible") continue
        const project = yield* projects.get(rawMeta.projectId).pipe(
          Effect.mapError((error) => new SessionOperationFailed({
            operation: `resolve project for session ${id}`,
            reason: String(error),
          })),
        )
        metas.push(storedMetaToProtocol(rawMeta, project.sourceDirectory))
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
        const promoted = yield* storage.sessions.updateMeta(sessionId, (current) => {
          const now = new Date().toISOString()
          return {
            ...(current ?? existing),
            visibility: "visible",
            updated: now,
          }
        }).pipe(
          Effect.mapError(toSessionErrorFromPlatform(`promote draft ${sessionId}`))
        )
        yield* publishChange
        return promoted
      }),

      listDraftSessionIds: Effect.fn("acn.session-store.list-draft-session-ids")(function* () {
        const ids = yield* storage.sessions.listSessionIds().pipe(
          Effect.mapError(toSessionErrorFromPlatform("list draft sessions")),
        )
        const draftIds: string[] = []
        for (const id of ids) {
          const meta = yield* readMetaForListing(id)
          if (meta?.visibility === "draft") draftIds.push(id)
        }
        return draftIds
      }),

      listProtocolMetas: Effect.fn("acn.session-store.list-protocol-metas")(function* (options) {
        const cwd = options?.cwd ? resolve(options.cwd) : null
        const projectId = options?.projectId ?? null
        const includeClosed = options?.includeClosed ?? true
        const query = options?.query?.trim().toLowerCase() ?? ""
        const cursor = options?.cursor ? decodeSessionCursor(options.cursor) : null
        const limit = clampSessionPageLimit(options?.limit)
        const metas = yield* readAllProtocolMetas()
        const matchingProjectIds = query
          ? new Set(
              (yield* projects.list())
                .filter((project) => project.name.toLowerCase().includes(query))
                .map((project) => project.projectId),
            )
          : new Set<ProjectId>()
        const filtered = metas
          .filter((meta) => !cwd || resolve(meta.cwd) === cwd)
          .filter((meta) => !projectId || meta.projectId === projectId)
          .filter((meta) => includeClosed || meta.sidebarOpen)
          .filter((meta) =>
            !query ||
            matchingProjectIds.has(meta.projectId) ||
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

      listAllProtocolMetas: readAllProtocolMetas,

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
        yield* publishChange
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
        const cwd = yield* projects.resolveSourceDirectory(meta.projectId).pipe(
          Effect.mapError((error) => new SessionOperationFailed({
            operation: `resolve project for session ${sessionId}`,
            reason: String(error),
          })),
        )
        return {
          cwd,
          projectRoot: cwd,
          scratchpadPath: storage.sessions.paths.sessionScratchpad(sessionId),
        }
      }),

      ensureProjectForCwd: Effect.fn("acn.session-store.ensure-project-for-cwd")(function* (cwd) {
        const project = yield* projects.ensureForSourceDirectory(cwd).pipe(
          Effect.mapError((error) => new SessionOperationFailed({
            operation: `resolve project for ${cwd}`,
            reason: String(error),
          })),
        )
        return project.projectId
      }),

      resolveProjectSource: Effect.fn("acn.session-store.resolve-project-source")(function* (
        projectId,
      ) {
        return yield* projects.resolveSourceDirectory(projectId).pipe(
          Effect.mapError((error) => new SessionOperationFailed({
            operation: `resolve project ${projectId}`,
            reason: String(error),
          })),
        )
      }),

      setSidebarOpen: Effect.fn("acn.session-store.set-sidebar-open")(function* (sessionId, open) {
        const existing = yield* readMeta(sessionId)
        if (!existing) return yield* new SessionNotFound({ sessionId })
        if (existing.sidebarOpen === open) {
          const unchanged = yield* readProtocolMeta(sessionId)
          if (!unchanged) return yield* new SessionNotFound({ sessionId })
          return unchanged
        }
        yield* storage.sessions.updateMeta(sessionId, (current) => ({
          ...(current ?? existing),
          sidebarOpen: open,
        })).pipe(
          Effect.mapError(toSessionErrorFromPlatform(`set sidebar membership ${sessionId}`)),
        )
        yield* publishChange
        const updated = yield* readProtocolMeta(sessionId)
        if (!updated) return yield* new SessionNotFound({ sessionId })
        return updated
      }),

      changes: Stream.fromPubSub(changes),
    }
  }),
)
