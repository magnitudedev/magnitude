/**
 * Sessions list hook — shared between web, desktop, and CLI.
 *
 * ListSessions query. Both apps use this identically.
 * Takes optional `{ cwd?, query?, cursor?, limit? }` params.
 */
import { Option } from "effect"
import { useMemo } from "react"
import { useAtomValue, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import type { SessionMetadata } from "@magnitudedev/sdk"
import type { ProjectId } from "@magnitudedev/sdk"

export interface UseSessionsListParams {
  /** Filter by CWD. If undefined, lists all sessions across all CWDs. */
  cwd?: string
  /** Filter by durable project identity. */
  projectId?: ProjectId
  /** Include sessions collapsed out of the sidebar. */
  includeClosed?: boolean
  /** Search title and working directory. */
  query?: string
  /** Cursor returned by the previous page. */
  cursor?: string
  /** Limit the number of results */
  limit?: number
}

export interface UseSessionsListResult {
  /** Whether the query is loading */
  loading: boolean
  /** A list-level failure. Individual unreadable sessions are omitted by ACN. */
  error: string | null
  /** Sessions list (empty during loading/failure) */
  sessions: SessionMetadata[]
  nextCursor: string | null
  hasMore: boolean
}

export function useSessionsList(params?: UseSessionsListParams): UseSessionsListResult {
  const client = useAgentClient()

  const sessionsAtom = useMemo(
    () =>
      client.rpc.query(
        "ListSessions",
        {
          cwd: params?.cwd !== undefined ? Option.some(params.cwd) : Option.none(),
          projectId: params?.projectId !== undefined
            ? Option.some(params.projectId)
            : Option.none(),
          includeClosed: params?.includeClosed ?? true,
          query: params?.query !== undefined ? Option.some(params.query) : Option.none(),
          cursor: params?.cursor !== undefined ? Option.some(params.cursor) : Option.none(),
          ...(params?.limit !== undefined ? { limit: params.limit } : {}),
        },
        { reactivityKeys: ["sessions"] },
      ),
    [client, params?.cwd, params?.projectId, params?.includeClosed, params?.query, params?.cursor, params?.limit],
  )

  const result = useAtomValue(sessionsAtom)

  const loading = Result.isInitial(result)
  const error = Result.isFailure(result) ? "Failed to load conversations." : null
  const sessions = Result.match(result, {
    onInitial: (): SessionMetadata[] => [],
    onFailure: (): SessionMetadata[] => [],
    onSuccess: (s) => [...s.value.items],
  })
  const nextCursor = Result.match(result, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (s) => {
      const cursor = s.value.nextCursor
      return cursor._tag === "Some" ? cursor.value : null
    },
  })
  const hasMore = Result.match(result, {
    onInitial: () => false,
    onFailure: () => false,
    onSuccess: (s) => s.value.hasMore,
  })

  return { loading, error, sessions, nextCursor, hasMore }
}
