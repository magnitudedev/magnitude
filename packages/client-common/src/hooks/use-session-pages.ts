/**
 * Paginated session list over `SessionInspector` pages.
 *
 * The first page is one reactive query; "Show more" explicitly requests
 * continuation pages (optionally with a different page size — cursors
 * fingerprint predicates, not limits). `loadAll` follows cursors to
 * exhaustion through plain read RPCs for select-all flows. Nothing here
 * auto-loads on scroll or issues a mutation.
 */
import { useCallback, useMemo } from "react"
import { Effect, Option } from "effect"
import { useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import type {
  DirectoryPath,
  SessionArchiveFilter,
  SessionMetadata,
  SessionPageCursor,
  SessionPinFilter,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { appendRequestedPage, makePageSet } from "../data/paginated-query"
import { sessionsToRecentChats, type RecentChat } from "../data/recent-chats"

const DEFAULT_SESSION_PAGE_SIZE = 50

export interface UseSessionPagesParams {
  /** Exact working-directory predicate (a Project's cwd). */
  readonly cwd?: DirectoryPath
  readonly archive?: SessionArchiveFilter
  readonly pin?: SessionPinFilter
  /** Server-side search over session title and cwd. */
  readonly query?: string
  readonly pageSize?: number
}

export interface UseSessionPagesResult {
  readonly sessions: ReadonlyArray<RecentChat>
  readonly loading: boolean
  readonly loadingMore: boolean
  readonly error: boolean
  readonly hasMore: boolean
  readonly loadMore: (limit?: number) => void
  /** Follows cursors to completion; used by select-all flows. */
  readonly loadAll: () => Promise<ReadonlyArray<RecentChat>>
}

export function useSessionPages(params?: UseSessionPagesParams): UseSessionPagesResult {
  const client = useAgentClient()
  const cwd = params?.cwd
  const archive = params?.archive ?? "active"
  const pin = params?.pin ?? "all"
  const query = params?.query
  const pageSize = params?.pageSize ?? DEFAULT_SESSION_PAGE_SIZE

  const payload = useCallback((cursor: Option.Option<SessionPageCursor>, limit: number) => ({
    cwd: Option.fromNullable(cwd),
    archive,
    pin,
    query: Option.fromNullable(query),
    cursor,
    limit,
  }), [cwd, archive, pin, query])

  const pageSet = useMemo(() => makePageSet<SessionMetadata, SessionPageCursor>({
    firstPage: client.rpc.query(
      "ListSessions",
      payload(Option.none(), pageSize),
      { reactivityKeys: ["sessions"] },
    ),
    continuationPage: ({ cursor, limit }) => client.rpc.query(
      "ListSessions",
      payload(Option.some(cursor), limit ?? pageSize),
      { reactivityKeys: ["sessions"] },
    ),
    itemKey: (session) => session.sessionId,
  }), [client, payload, pageSize])

  const snapshot = useAtomValue(pageSet.snapshot)
  const setRequested = useAtomSet(pageSet.requestedPages)
  const nextCursor = snapshot.nextCursor
  const loadMore = useCallback((limit?: number) => {
    if (nextCursor === null) return
    setRequested((current) => appendRequestedPage(current, nextCursor, limit))
  }, [nextCursor, setRequested])

  const loadAllAtom = useMemo(() => client.rpc.runtime.fn<void>()(() =>
    Effect.gen(function* () {
      const rpc = yield* client.rpc
      const all: SessionMetadata[] = []
      let cursor = Option.none<SessionPageCursor>()
      while (true) {
        const page = yield* rpc("ListSessions", payload(cursor, 100))
        all.push(...page.items)
        if (Option.isNone(page.nextCursor)) break
        cursor = Option.some(page.nextCursor.value)
      }
      return sessionsToRecentChats(all)
    }),
  ), [client, payload])
  const runLoadAll = useAtomSet(loadAllAtom, { mode: "promise" })
  const loadAll = useCallback(() => runLoadAll(), [runLoadAll])

  const sessions = useMemo(() => sessionsToRecentChats(snapshot.items), [snapshot.items])

  return {
    sessions,
    loading: snapshot.loading,
    loadingMore: snapshot.loadingMore,
    error: snapshot.error,
    hasMore: snapshot.hasMore,
    loadMore,
    loadAll,
  }
}
