/**
 * Shared paginated sessions hook.
 *
 * Builds on `useSessionsList` for the first page and accumulates subsequent
 * pages via a writable atom. The accumulation atom is recreated when the
 * request identity changes — that recreation IS the reset, no
 * ref-diff or useEffect needed.
 */
import { useCallback, useMemo } from "react"
import { Option } from "effect"
import { Atom, useAtom, useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { useSessionsList, type UseSessionsListParams } from "./use-sessions-list"
import { sessionsToRecentChats } from "../data/recent-chats"
import type { SessionMetadata } from "@magnitudedev/sdk"
import type { ProjectId, SessionArchiveFilter } from "@magnitudedev/sdk"
import type { RecentChat } from "../data/recent-chats"

export interface UsePaginatedSessionsParams {
  /** Filter by CWD. */
  cwd?: string
  projectId?: ProjectId
  archiveFilter?: SessionArchiveFilter
  prioritizePinned?: boolean
  /** Search title and working directory. */
  query?: string
  /** Number of items per page. */
  pageSize?: number
}

export interface UsePaginatedSessionsResult {
  /** Combined sessions from first page + accumulated pages. */
  sessions: RecentChat[]
  /** Whether the first page is loading and no sessions are available yet. */
  loading: boolean
  /** A list-level failure while loading the first page. */
  error: string | null
  /** Whether a subsequent page is currently loading. */
  loadingMore: boolean
  /** Whether more pages can be loaded. */
  hasMore: boolean
  /** Total sessions matching the filter, independent of loaded pages. */
  totalCount: number
  /** Load the next page. */
  loadMore: () => void
  /** Load and return every session matching the current filters. */
  loadAll: () => Promise<RecentChat[]>
}

/**
 * Design note: viewport-fill auto-loading.
 *
 * This hook intentionally exposes only the core pagination primitives
 * (first-page query, accumulated pages, loadMore, and loadAll). A reusable
 * viewport-fill layer can be built on top without changing this hook:
 *
 *   const { sessions, hasMore, loadingMore, loadMore } = usePaginatedSessions(...)
 *   useViewportFill({
 *     ref: scrollableRef,
 *     hasMore,
 *     loading: loadingMore,
 *     onLoadMore: loadMore,
 *   })
 *
 * The layer would measure the scrollable container's viewport height and
 * content height on mount and after every render, and call `loadMore` while
 * `hasMore && !loading && contentHeight < viewportHeight`. This is the same
 * sufficiency logic the chat timeline uses via
 * `TimelineScrollController.reconcileRootShape()`.
 *
 * It is not added here because the measurement primitive differs between
 * surfaces (OpenTUI scrollbox vs. DOM element vs. web CustomScrollArea),
 * so the adapter belongs in its own hook.
 */

interface AccumulationState {
  firstPageIdentity: string
  extraSessions: SessionMetadata[]
  nextCursor: string | null
  hasMore: boolean
}

const uniqueSessions = (sessions: readonly SessionMetadata[]): SessionMetadata[] => {
  const seen = new Set<string>()
  return sessions.filter((session) => {
    if (seen.has(session.sessionId)) return false
    seen.add(session.sessionId)
    return true
  })
}

export function usePaginatedSessions(params?: UsePaginatedSessionsParams): UsePaginatedSessionsResult {
  const client = useAgentClient()
  const listSessionsAtom = useMemo(() => client.rpc.mutation("ListSessions"), [client])
  const listSessionsResult = useAtomValue(listSessionsAtom)
  const listSessionsMutation = useAtomSet(listSessionsAtom, { mode: "promise" })
  const loadingMore = Result.isWaiting(listSessionsResult)

  const firstPage = useSessionsList({
    cwd: params?.cwd,
    projectId: params?.projectId,
    archiveFilter: params?.archiveFilter,
    prioritizePinned: params?.prioritizePinned,
    query: params?.query,
    limit: params?.pageSize ?? 50,
  })

  const firstPageIdentity = firstPage.sessions.map((session) => [
    session.sessionId,
    session.archived ? "1" : "0",
    Option.getOrNull(session.pinnedAt) ?? "",
    session.updatedAt,
  ].join(":"))
    .concat(
      firstPage.nextCursor ?? "",
      firstPage.hasMore ? "1" : "0",
      String(firstPage.totalCount),
    )
    .join("\0")
  const accumulationAtom = useMemo(
    () =>
      Atom.make<AccumulationState>({
        firstPageIdentity,
        extraSessions: [],
        nextCursor: firstPage.nextCursor,
        hasMore: firstPage.hasMore,
      }),
    [params?.cwd, params?.projectId, params?.archiveFilter, params?.prioritizePinned, params?.query],
  )
  const [accumulation, setAccumulation] = useAtom(accumulationAtom)
  const currentAccumulation = accumulation.firstPageIdentity === firstPageIdentity
    ? accumulation
    : {
        firstPageIdentity,
        extraSessions: [],
        nextCursor: firstPage.nextCursor,
        hasMore: firstPage.hasMore,
      }
  const nextCursor = currentAccumulation.extraSessions.length === 0
    ? firstPage.nextCursor
    : currentAccumulation.nextCursor
  const hasMore = currentAccumulation.extraSessions.length === 0
    ? firstPage.hasMore
    : currentAccumulation.hasMore

  const loadMore = useCallback(() => {
    if (loadingMore || !hasMore || !nextCursor) return

    const cursor = nextCursor

    void listSessionsMutation({
      payload: {
        cwd: params?.cwd !== undefined ? Option.some(params.cwd) : Option.none(),
        projectId: params?.projectId !== undefined
          ? Option.some(params.projectId)
          : Option.none(),
        archiveFilter: params?.archiveFilter ?? "active",
        prioritizePinned: params?.prioritizePinned ?? false,
        query: params?.query !== undefined ? Option.some(params.query) : Option.none(),
        cursor: Option.some(cursor),
        ...(params?.pageSize !== undefined ? { limit: params.pageSize } : {}),
      },
      reactivityKeys: ["sessions"],
    })
      .then((page) => {
        setAccumulation((previous) => ({
          firstPageIdentity,
          extraSessions: [
            ...(previous.firstPageIdentity === firstPageIdentity ? previous.extraSessions : []),
            ...page.items,
          ],
          nextCursor: page.nextCursor._tag === "Some" ? page.nextCursor.value : null,
          hasMore: page.hasMore,
        }))
      })
      .catch(() => {
        // The mutation Result is the authoritative failure state; there is no
        // local error mirror to update here.
      })
  }, [
    listSessionsMutation,
    loadingMore,
    hasMore,
    nextCursor,
    params?.cwd,
    params?.projectId,
    params?.archiveFilter,
    params?.prioritizePinned,
    params?.query,
    params?.pageSize,
    firstPageIdentity,
    setAccumulation,
  ])

  const loadAll = useCallback(async (): Promise<RecentChat[]> => {
    const loadedSessions = [...firstPage.sessions]
    let cursor = firstPage.nextCursor
    let more = firstPage.hasMore

    while (more && cursor) {
      const loadedPage = await listSessionsMutation({
        payload: {
          cwd: params?.cwd !== undefined ? Option.some(params.cwd) : Option.none(),
          projectId: params?.projectId !== undefined
            ? Option.some(params.projectId)
            : Option.none(),
          archiveFilter: params?.archiveFilter ?? "active",
          prioritizePinned: params?.prioritizePinned ?? false,
          query: params?.query !== undefined ? Option.some(params.query) : Option.none(),
          cursor: Option.some(cursor),
          ...(params?.pageSize !== undefined ? { limit: params.pageSize } : {}),
        },
        reactivityKeys: ["sessions"],
      })
      loadedSessions.push(...loadedPage.items)
      cursor = Option.getOrNull(loadedPage.nextCursor)
      more = loadedPage.hasMore
    }

    const uniqueLoadedSessions = uniqueSessions(loadedSessions)
    setAccumulation({
      firstPageIdentity,
      extraSessions: uniqueLoadedSessions.slice(firstPage.sessions.length),
      nextCursor: cursor,
      hasMore: more,
    })
    return sessionsToRecentChats(uniqueLoadedSessions)
  }, [
    firstPage.sessions,
    firstPage.nextCursor,
    firstPage.hasMore,
    firstPageIdentity,
    listSessionsMutation,
    params?.cwd,
    params?.projectId,
    params?.archiveFilter,
    params?.prioritizePinned,
    params?.query,
    params?.pageSize,
    setAccumulation,
  ])

  const sessions = sessionsToRecentChats(uniqueSessions([
    ...firstPage.sessions,
    ...currentAccumulation.extraSessions,
  ]))

  const loading = firstPage.loading && sessions.length === 0

  return {
    sessions,
    loading,
    error: firstPage.error,
    loadingMore,
    hasMore,
    totalCount: firstPage.totalCount,
    loadMore,
    loadAll,
  }
}
