/**
 * Usage state hook — shared between web, desktop, and CLI.
 *
 * GetBalance query. Both apps use this identically.
 */
import { useMemo, useState } from "react"
import { useAtomValue, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import type { BalanceResponse, UsagePeriod } from "@magnitudedev/sdk"

export interface UseUsageStateResult {
  /** Whether the query is loading */
  loading: boolean
  /** Error message if the query failed */
  error: string | null
  /** Balance data if the query succeeded */
  data: BalanceResponse | null
  /** Currently selected period */
  period: UsagePeriod
  /** Change the selected period */
  setPeriod: (period: UsagePeriod) => void
}

export function useUsageState(): UseUsageStateResult {
  const client = useAgentClient()
  const [period, setPeriod] = useState<UsagePeriod>("24h")

  const balanceAtom = useMemo(
    () => client.query("GetBalance", { period }),
    [client, period],
  )
  const result = useAtomValue(balanceAtom)

  const loading = Result.isInitial(result)
  const error = Result.isFailure(result) ? "Failed to load usage data." : null
  const data = Result.isSuccess(result) ? result.value : null

  return { loading, error, data, period, setPeriod }
}
