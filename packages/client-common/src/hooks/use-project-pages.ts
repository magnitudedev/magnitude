/**
 * Paginated Project list. First page loads through one reactive query;
 * "Show more" explicitly requests continuation pages. Never auto-loads.
 */
import { useCallback, useMemo } from "react"
import { Option } from "effect"
import { useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import type { Project, ProjectPageCursor } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { appendRequestedPage, makePageSet } from "../data/paginated-query"

const DEFAULT_PROJECT_PAGE_SIZE = 20

export interface UseProjectPagesParams {
  readonly includeRemoved?: boolean
  readonly pageSize?: number
}

export interface UseProjectPagesResult {
  readonly projects: ReadonlyArray<Project>
  readonly loading: boolean
  readonly loadingMore: boolean
  readonly error: boolean
  readonly hasMore: boolean
  readonly loadMore: (limit?: number) => void
}

export function useProjectPages(params?: UseProjectPagesParams): UseProjectPagesResult {
  const client = useAgentClient()
  const includeRemoved = params?.includeRemoved ?? false
  const pageSize = params?.pageSize ?? DEFAULT_PROJECT_PAGE_SIZE

  const pageSet = useMemo(() => makePageSet<Project, ProjectPageCursor>({
    firstPage: client.rpc.query(
      "ListProjects",
      { includeRemoved, cursor: Option.none<ProjectPageCursor>(), limit: pageSize },
      { reactivityKeys: ["projects"] },
    ),
    continuationPage: ({ cursor, limit }) => client.rpc.query(
      "ListProjects",
      { includeRemoved, cursor: Option.some(cursor), limit: limit ?? pageSize },
      { reactivityKeys: ["projects"] },
    ),
    itemKey: (project) => project.projectId,
  }), [client, includeRemoved, pageSize])

  const snapshot = useAtomValue(pageSet.snapshot)
  const setRequested = useAtomSet(pageSet.requestedPages)
  const nextCursor = snapshot.nextCursor
  const loadMore = useCallback((limit?: number) => {
    if (nextCursor === null) return
    setRequested((current) => appendRequestedPage(current, nextCursor, limit))
  }, [nextCursor, setRequested])

  return {
    projects: snapshot.items,
    loading: snapshot.loading,
    loadingMore: snapshot.loadingMore,
    error: snapshot.error,
    hasMore: snapshot.hasMore,
    loadMore,
  }
}
