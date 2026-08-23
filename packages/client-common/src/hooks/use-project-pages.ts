/**
 * Paginated Project list. First page loads through one query;
 * "Show more" explicitly requests continuation pages. Never auto-loads.
 * Pages stay fresh through the connection's change pokes.
 */
import { useCallback, useMemo } from "react"
import { Option } from "effect"
import { Atom, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Key } from "@magnitudedev/effect-query"
import type { Project, ProjectPageCursor } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { appendRequestedPage, makePageSet, type RequestedPage } from "../data/paginated-query"

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

  const pageSet = useMemo(() => {
    const page = (cursor: Option.Option<ProjectPageCursor>, limit: number) =>
      Atom.make((get) => get(client.Projects.ListProjects({ includeRemoved, cursor, limit })).result)
    const continuations = new Map<string, ReturnType<typeof page>>()
    return makePageSet<Project, ProjectPageCursor>({
      firstPage: page(Option.none(), pageSize),
      continuationPage: (request: RequestedPage<ProjectPageCursor>) => {
        const key = Key.canonical(request)
        const existing = continuations.get(key)
        if (existing !== undefined) return existing
        const created = page(Option.some(request.cursor), request.limit ?? pageSize)
        continuations.set(key, created)
        return created
      },
      itemKey: (project) => project.projectId,
    })
  }, [client, includeRemoved, pageSize])

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
