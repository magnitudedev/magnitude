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
import type { ListSessionsResult, SessionMetadata } from "@magnitudedev/sdk"

export interface UseSessionsListParams {
  /** Filter by CWD. If undefined, lists all sessions across all CWDs. */
  cwd?: string
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
  /** Sessions list (empty during loading/failure) */
  sessions: SessionMetadata[]
  nextCursor: string | null
  hasMore: boolean
}

export function useSessionsList(params?: UseSessionsListParams): UseSessionsListResult {
  const client = useAgentClient()

  const sessionsAtom = useMemo(
    () =>
      client.query(
        "ListSessions",
        {
          cwd: params?.cwd !== undefined ? Option.some(params.cwd) : Option.none(),
          query: params?.query !== undefined ? Option.some(params.query) : Option.none(),
          cursor: params?.cursor !== undefined ? Option.some(params.cursor) : Option.none(),
          ...(params?.limit !== undefined ? { limit: params.limit } : {}),
        },
        { reactivityKeys: ["sessions"] },
      ),
    [client, params?.cwd, params?.query, params?.cursor, params?.limit],
  )

  const result = useAtomValue(sessionsAtom)

  const loading = Result.isInitial(result)
  const sessions = Result.match(result, {
    onInitial: () => [] as SessionMetadata[],
    onFailure: () => [] as SessionMetadata[],
    onSuccess: (s) => [...(s.value as ListSessionsResult).items],
  })
  const nextCursor = Result.match(result, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (s) => {
      const cursor = (s.value as ListSessionsResult).nextCursor
      return cursor._tag === "Some" ? cursor.value : null
    },
  })
  const hasMore = Result.match(result, {
    onInitial: () => false,
    onFailure: () => false,
    onSuccess: (s) => (s.value as ListSessionsResult).hasMore,
  })

  return { loading, sessions, nextCursor, hasMore }
}
