import { useMemo } from "react"
import { useAtomValue, Result } from "@effect-atom/atom-react"
import type { ProjectSummary } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"

export interface UseProjectsResult {
  readonly projects: ReadonlyArray<ProjectSummary>
  readonly revealKind: "finder" | "folder" | "unsupported"
  readonly loading: boolean
  readonly error: string | null
}

export function useProjects(includeRemoved = false): UseProjectsResult {
  const client = useAgentClient()
  const query = useMemo(
    () => client.rpc.query(
      "ListProjects",
      { includeRemoved },
      { reactivityKeys: ["projects", "sessions"] },
    ),
    [client, includeRemoved],
  )
  const result = useAtomValue(query)
  return {
    projects: Result.match(result, {
      onInitial: () => [],
      onFailure: (failure) => failure.previousSuccess._tag === "Some"
        ? failure.previousSuccess.value.value.projects
        : [],
      onSuccess: (success) => success.value.projects,
    }),
    revealKind: Result.match(result, {
      onInitial: () => "unsupported" as const,
      onFailure: (failure) => failure.previousSuccess._tag === "Some"
        ? failure.previousSuccess.value.value.revealKind
        : "unsupported" as const,
      onSuccess: (success) => success.value.revealKind,
    }),
    loading: Result.isInitial(result),
    error: Result.isFailure(result) ? "Failed to load projects." : null,
  }
}
