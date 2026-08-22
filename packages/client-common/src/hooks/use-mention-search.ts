/**
 * Mention search as an imperative read of the `SearchMentions` query.
 */
import { useMemo } from "react"
import { useAtomSet } from "@effect-atom/atom-react"
import { QueryClient } from "@magnitudedev/effect-query"
import { Files } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import type { MentionSearchClient } from "./use-file-mentions"


type MentionSearchInput = Parameters<MentionSearchClient["searchMentions"]>[0]

export function useMentionSearchClient(): MentionSearchClient {
  const client = useAgentClient()
  const searchAtom = useMemo(() => client.runtime.fn<MentionSearchInput>()(
    (payload) => QueryClient.fetch(Files.SearchMentions, payload),
  ), [client])
  const search = useAtomSet(searchAtom, { mode: "promise" })
  return useMemo<MentionSearchClient>(() => ({
    searchMentions(payload) {
      return search({
        cwd: payload.cwd,
        query: payload.query,
        ...(payload.limit !== undefined ? { limit: payload.limit } : {}),
        ...(payload.visibleLimit !== undefined ? { visibleLimit: payload.visibleLimit } : {}),
        ...(payload.includeRecent !== undefined ? { includeRecent: payload.includeRecent } : {}),
      })
    },
  }), [search])
}
