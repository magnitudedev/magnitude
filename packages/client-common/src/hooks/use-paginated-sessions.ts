/**
 * Shared paginated sessions hook.
 *
 * Builds on `useSessionsList` for the first page and accumulates subsequent
 * pages via a writable atom. The accumulation atom is recreated when the
 * request key (cwd + query) changes — that recreation IS the reset, no
 * ref-diff or useEffect needed.
 */
import { useCallback, useMemo } from "react"
import { Option } from "effect"
import { Atom, useAtom, useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { useSessionsList, type UseSessionsListParams } from "./use-sessions-list"
import { sessionsToRecentChats } from "../data/recent-chats"
import type { SessionMetadata } from "@magnitudedev/sdk"
import type { RecentChat } from "../data/recent-chats"

export interface UsePaginatedSessionsParams {
  /** Filter by CWD. */
  cwd?: string
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
  /** Whether a subsequent page is currently loading. */
  loadingMore: boolean
  /** Whether more pages can be loaded. */
  hasMore: boolean
  /** Load the next page. */
  loadMore: () => void
}

/**
 * Design note: viewport-fill auto-loading.
 *
 * This hook intentionally exposes only the core pagination primitive
 * (first-page query + accumulated pages + loadMore). A reusable
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
  const listSessionsAtom = useMemo(() => client.mutation("ListSessions"), [client])
  const listSessionsResult = useAtomValue(listSessionsAtom)
  const listSessionsMutation = useAtomSet(listSessionsAtom, { mode: "promise" })
  const loadingMore = Result.isWaiting(listSessionsResult)

  const firstPage = useSessionsList({
    cwd: params?.cwd,
    query: params?.query,
    limit: params?.pageSize ?? 50,
  })

  const accumulationAtom = useMemo(
    () =>
      Atom.make<AccumulationState>({
        extraSessions: [],
        nextCursor: firstPage.nextCursor,
        hasMore: firstPage.hasMore,
      }),
    [params?.cwd, params?.query],
  )
  const [accumulation, setAccumulation] = useAtom(accumulationAtom)
  const nextCursor = accumulation.extraSessions.length === 0
    ? firstPage.nextCursor
    : accumulation.nextCursor
  const hasMore = accumulation.extraSessions.length === 0
    ? firstPage.hasMore
    : accumulation.hasMore

  const loadMore = useCallback(() => {
    if (loadingMore || !hasMore || !nextCursor) return

    const cursor = nextCursor

    void listSessionsMutation({
      payload: {
        cwd: params?.cwd !== undefined ? Option.some(params.cwd) : Option.none(),
        query: params?.query !== undefined ? Option.some(params.query) : Option.none(),
        cursor: Option.some(cursor),
        ...(params?.pageSize !== undefined ? { limit: params.pageSize } : {}),
      },
      reactivityKeys: ["sessions"],
    })
      .then((page) => {
        setAccumulation((prev) => ({
          ...prev,
          extraSessions: [...prev.extraSessions, ...page.items],
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
    params?.query,
    params?.pageSize,
    setAccumulation,
  ])

  const sessions = sessionsToRecentChats(uniqueSessions([
    ...firstPage.sessions,
    ...accumulation.extraSessions,
  ]))

  const loading = firstPage.loading && sessions.length === 0

  return {
    sessions,
    loading,
    loadingMore,
    hasMore,
    loadMore,
  }
}
