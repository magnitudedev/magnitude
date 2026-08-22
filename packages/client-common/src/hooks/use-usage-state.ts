/**
 * Usage state hook — shared between web, desktop, and CLI.
 *
 * GetCloudUsage query. Both apps use this identically.
 */
import { useMemo, useState } from "react"
import { Atom, useAtomValue, Result } from "@effect-atom/atom-react"
import { Configuration, type CloudUsageResponse, type UsagePeriod } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"


export interface UseUsageStateResult {
  /** Whether the query is loading */
  loading: boolean
  /** Error message if the query failed */
  error: string | null
  /** Cloud subscription and usage data if the query succeeded */
  data: CloudUsageResponse | null
  /** Currently selected period */
  period: UsagePeriod
  /** Change the selected period */
  setPeriod: (period: UsagePeriod) => void
}

export function useUsageState(): UseUsageStateResult {
  const client = useAgentClient()
  const [period, setPeriod] = useState<UsagePeriod>("24h")

  const usageAtom = useMemo(
    () => Atom.make((get) => get(client.query(Configuration.GetCloudUsage, { period })).result),
    [client, period],
  )
  const result = useAtomValue(usageAtom)

  const loading = Result.isInitial(result)
  const error = Result.isFailure(result) ? "Failed to load usage data." : null
  const data = Result.isSuccess(result) ? result.value : null

  return { loading, error, data, period, setPeriod }
}
