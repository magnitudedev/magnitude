import {
  DirectoryPathSchema,
  InvalidDirectoryPageCursor,
  InvalidSessionPageCursor,
  SessionInspectionUnavailable,
  SessionMetadataUnreadable,
  SessionNotFound,
  SessionPageCursorSchema,
  type DirectoryPageCursor,
  DirectoryPageCursorSchema,
  type RecentDirectory,
  type RecentDirectoryPage,
  type RecentDirectoryPageRequest,
  type SessionMetadata,
  type SessionPage,
  type SessionPageCursor,
  type SessionPageRequest,
  type SessionPinFilter,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage, type StoredSessionMeta } from "@magnitudedev/storage"
import { Context, Effect, Either, Layer, Option, Schema, Stream } from "effect"

export interface SessionInspector {
  readonly get: (
    sessionId: string,
  ) => Effect.Effect<SessionMetadata, SessionNotFound | SessionMetadataUnreadable>
  readonly page: (
    request: SessionPageRequest,
  ) => Effect.Effect<SessionPage, InvalidSessionPageCursor | SessionInspectionUnavailable>
  readonly recentDirectories: (
    request: RecentDirectoryPageRequest,
  ) => Effect.Effect<RecentDirectoryPage, InvalidDirectoryPageCursor | SessionInspectionUnavailable>
  readonly changes: Stream.Stream<void>
}

export const SessionInspector = Context.GenericTag<SessionInspector>("acn/SessionInspector")

/**
 * Convert one stored record to protocol metadata. A record whose timestamps do
 * not parse is unreadable — corrupt data is never coerced into a ranking.
 */
export const decodeSessionMetadata = (
  meta: StoredSessionMeta,
): Either.Either<SessionMetadata, SessionMetadataUnreadable> => {
  const createdAt = Date.parse(meta.created)
  const updatedAt = Date.parse(meta.updated)
  if (!Number.isFinite(createdAt) || !Number.isFinite(updatedAt)) {
    return Either.left(new SessionMetadataUnreadable({ sessionId: meta.sessionId }))
  }
  const pinnedAt = Option.map(meta.pinnedAt, Date.parse)
  if (Option.isSome(pinnedAt) && !Number.isFinite(pinnedAt.value)) {
    return Either.left(new SessionMetadataUnreadable({ sessionId: meta.sessionId }))
  }
  return Either.right({
    sessionId: meta.sessionId,
    title: meta.chatName,
    cwd: meta.workingDirectory,
    archived: meta.archived,
    pinnedAt,
    createdAt,
    updatedAt,
    messageCount: meta.messageCount,
    lastMessage: meta.lastMessage,
  })
}

const normalizedQuery = (request: SessionPageRequest): string =>
  Option.getOrElse(request.query, () => "").trim().toLowerCase()

/** Predicate identity carried by session cursors; page size deliberately excluded. */
const fingerprintOf = (request: SessionPageRequest): string => JSON.stringify({
  cwd: Option.getOrNull(request.cwd),
  archive: request.archive,
  pin: request.pin,
  query: normalizedQuery(request),
})

const SessionCursorContent = Schema.compose(
  Schema.StringFromBase64Url,
  Schema.parseJson(Schema.Struct({
    fingerprint: Schema.String,
    pinnedAt: Schema.NullOr(Schema.Number),
    updatedAt: Schema.Number,
    sessionId: Schema.String,
  })),
)

interface SessionCursorKey {
  readonly pinnedAt: number | null
  readonly updatedAt: number
  readonly sessionId: string
}

const decodeSessionCursor = (
  request: SessionPageRequest,
  cursor: SessionPageCursor,
): Effect.Effect<SessionCursorKey, InvalidSessionPageCursor> =>
  Schema.decodeUnknown(SessionCursorContent)(cursor).pipe(
    Effect.mapError(() => new InvalidSessionPageCursor()),
    Effect.filterOrFail(
      (content) => content.fingerprint === fingerprintOf(request),
      () => new InvalidSessionPageCursor(),
    ),
  )

const encodeSessionCursor = (
  request: SessionPageRequest,
  session: SessionMetadata,
): SessionPageCursor =>
  SessionPageCursorSchema.make(Schema.encodeSync(SessionCursorContent)({
    fingerprint: fingerprintOf(request),
    pinnedAt: Option.getOrNull(session.pinnedAt),
    updatedAt: session.updatedAt,
    sessionId: session.sessionId,
  }))

const byOrdering = (pin: SessionPinFilter) => (
  left: SessionMetadata,
  right: SessionMetadata,
): number => {
  if (pin === "pinned") {
    const pinnedDelta = Option.getOrElse(right.pinnedAt, () => 0)
      - Option.getOrElse(left.pinnedAt, () => 0)
    if (pinnedDelta !== 0) return pinnedDelta
  }
  return right.updatedAt - left.updatedAt || right.sessionId.localeCompare(left.sessionId)
}

const strictlyAfter = (
  session: SessionMetadata,
  cursor: SessionCursorKey,
  pin: SessionPinFilter,
): boolean => {
  if (pin === "pinned") {
    const sessionPinnedAt = Option.getOrElse(session.pinnedAt, () => 0)
    const cursorPinnedAt = cursor.pinnedAt ?? 0
    if (sessionPinnedAt !== cursorPinnedAt) return sessionPinnedAt < cursorPinnedAt
  }
  return session.updatedAt < cursor.updatedAt
    || (session.updatedAt === cursor.updatedAt && session.sessionId < cursor.sessionId)
}

const DirectoryCursorContent = Schema.compose(
  Schema.StringFromBase64Url,
  Schema.parseJson(Schema.Struct({
    lastActiveAt: Schema.Number,
    cwd: DirectoryPathSchema,
  })),
)

const decodeDirectoryCursor = (cursor: DirectoryPageCursor) =>
  Schema.decodeUnknown(DirectoryCursorContent)(cursor).pipe(
    Effect.mapError(() => new InvalidDirectoryPageCursor()),
  )

const encodeDirectoryCursor = (entry: RecentDirectory): DirectoryPageCursor =>
  DirectoryPageCursorSchema.make(Schema.encodeSync(DirectoryCursorContent)({
    lastActiveAt: entry.lastActiveAt,
    cwd: entry.cwd,
  }))

const CHUNK = 16

export const SessionInspectorLive: Layer.Layer<
  SessionInspector,
  never,
  MagnitudeStorage
> = Layer.effect(
  SessionInspector,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage

    const readVisible = (sessionId: string) => storage.sessions.readMeta(sessionId).pipe(
      Effect.map((meta) => meta?.visibility === "visible" ? meta : null),
    )

    const skipUnreadable = (sessionId: string, errorTag: string) =>
      Effect.logWarning("Skipping unreadable session metadata").pipe(
        Effect.annotateLogs({ sessionId, errorTag }),
        Effect.as(null),
      )

    /** Visible protocol metadata, or null when missing/draft/unreadable (logged). */
    const readForListing = (sessionId: string): Effect.Effect<SessionMetadata | null> =>
      readVisible(sessionId).pipe(
        Effect.flatMap((meta) => meta === null
          ? Effect.succeed(null)
          : Either.match(decodeSessionMetadata(meta), {
              onLeft: (error) => skipUnreadable(sessionId, error._tag),
              onRight: (session) => Effect.succeed<SessionMetadata | null>(session),
            })),
        Effect.catchAll((error) => skipUnreadable(sessionId, error._tag)),
      )

    const matchesFilters = (request: SessionPageRequest, session: SessionMetadata): boolean => {
      if (request.archive !== "all"
        && (request.archive === "archived") !== session.archived) return false
      if (request.pin !== "all"
        && (request.pin === "pinned") !== Option.isSome(session.pinnedAt)) return false
      const query = normalizedQuery(request)
      if (query.length === 0) return true
      return (session.title ?? "").toLowerCase().includes(query)
        || session.cwd.toLowerCase().includes(query)
    }

    /**
     * Exact-cwd paging trusts the cwd index's recency order (maintained
     * move-to-front under the metadata lock) and reads only enough records to
     * fill the page plus its continuation probe.
     */
    const pageByCwdIndex = Effect.fn("acn.session-inspector.page-by-cwd-index")(function* (
      request: SessionPageRequest,
      cwd: string,
      cursor: Option.Option<SessionCursorKey>,
    ) {
      const index = yield* storage.sessions.readCwdIndex(cwd).pipe(
        Effect.mapError(() => new SessionInspectionUnavailable()),
      )
      const ids = index?.sessionIds ?? []
      const survivors: SessionMetadata[] = []
      for (let offset = 0; offset < ids.length && survivors.length <= request.limit; offset += CHUNK) {
        const metas = yield* Effect.forEach(
          ids.slice(offset, offset + CHUNK),
          readForListing,
          { concurrency: CHUNK },
        )
        for (const session of metas) {
          if (session === null) continue
          if (session.cwd !== cwd) continue // stale legacy index entry
          if (!matchesFilters(request, session)) continue
          if (Option.isSome(cursor) && !strictlyAfter(session, cursor.value, request.pin)) continue
          survivors.push(session)
          if (survivors.length > request.limit) break
        }
      }
      return survivors
    })

    const pageGlobal = Effect.fn("acn.session-inspector.page-global")(function* (
      request: SessionPageRequest,
      cursor: Option.Option<SessionCursorKey>,
    ) {
      const ids = yield* storage.sessions.listSessionIds().pipe(
        Effect.mapError(() => new SessionInspectionUnavailable()),
      )
      const metas = yield* Effect.forEach(ids, readForListing, { concurrency: CHUNK })
      return metas
        .filter((session): session is SessionMetadata => session !== null)
        .filter((session) => matchesFilters(request, session))
        .sort(byOrdering(request.pin))
        .filter((session) => Option.match(cursor, {
          onNone: () => true,
          onSome: (key) => strictlyAfter(session, key, request.pin),
        }))
        .slice(0, request.limit + 1)
    })

    return SessionInspector.of({
      get: Effect.fn("acn.session-inspector.get")(function* (sessionId) {
        const meta = yield* readVisible(sessionId).pipe(
          Effect.mapError(() => new SessionMetadataUnreadable({ sessionId })),
        )
        if (meta === null) return yield* new SessionNotFound({ sessionId })
        return yield* decodeSessionMetadata(meta)
      }),
      page: Effect.fn("acn.session-inspector.page")(function* (request) {
        const cursor = yield* Option.match(request.cursor, {
          onNone: () => Effect.succeedNone,
          onSome: (value) => decodeSessionCursor(request, value).pipe(Effect.map(Option.some)),
        })
        const window = yield* Option.match(request.cwd, {
          onNone: () => pageGlobal(request, cursor),
          onSome: (cwd) => pageByCwdIndex(request, cwd, cursor),
        })
        const items = window.slice(0, request.limit)
        const last = items.at(-1)
        return {
          items,
          nextCursor: window.length > request.limit && last !== undefined
            ? Option.some(encodeSessionCursor(request, last))
            : Option.none(),
        }
      }),
      recentDirectories: Effect.fn("acn.session-inspector.recent-directories")(function* (
        request,
      ) {
        const cursor = yield* Option.match(request.cursor, {
          onNone: () => Effect.succeedNone,
          onSome: (value) => decodeDirectoryCursor(value).pipe(Effect.map(Option.some)),
        })
        const ids = yield* storage.sessions.listSessionIds().pipe(
          Effect.mapError(() => new SessionInspectionUnavailable()),
        )
        const metas = yield* Effect.forEach(ids, readForListing, { concurrency: CHUNK })
        const byCwd = new Map<string, RecentDirectory>()
        for (const session of metas) {
          if (session === null) continue
          const existing = byCwd.get(session.cwd)
          byCwd.set(session.cwd, {
            cwd: session.cwd,
            lastActiveAt: Math.max(existing?.lastActiveAt ?? 0, session.updatedAt),
            sessionCount: (existing?.sessionCount ?? 0) + 1,
          })
        }
        const ordered = [...byCwd.values()]
          .sort((left, right) =>
            right.lastActiveAt - left.lastActiveAt || right.cwd.localeCompare(left.cwd))
          .filter((entry) => Option.match(cursor, {
            onNone: () => true,
            onSome: (key) => entry.lastActiveAt < key.lastActiveAt
              || (entry.lastActiveAt === key.lastActiveAt && entry.cwd < key.cwd),
          }))
        const window = ordered.slice(0, request.limit + 1)
        const items = window.slice(0, request.limit)
        const last = items.at(-1)
        return {
          items,
          nextCursor: window.length > request.limit && last !== undefined
            ? Option.some(encodeDirectoryCursor(last))
            : Option.none(),
        }
      }),
      changes: storage.sessions.metadataChanges.pipe(Stream.as<void>(undefined)),
    })
  }),
)
