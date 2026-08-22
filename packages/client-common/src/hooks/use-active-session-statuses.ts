/**
 * Resident-session statuses: one stream-folded query (`StreamActiveSessionStatuses`).
 * The latest snapshot is the query's data; nothing is copied into client state.
 */
import { useMemo } from "react"
import { Atom, Result, useAtomValue } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
  Sessions,
  type ActiveSessionStatus,
  type ActiveSessionStatuses,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"

export type ActiveSessionStatusById = Readonly<Record<string, ActiveSessionStatus>>

const EMPTY: ActiveSessionStatusById = {}

const toStatusById = (snapshot: ActiveSessionStatuses): ActiveSessionStatusById => {
  const byId: Record<string, ActiveSessionStatus> = {}
  for (const status of snapshot.sessions) {
    byId[status.sessionId] = status
  }
  return byId
}

/** Live statuses by session id; empty until the first snapshot arrives. */
export function useActiveSessionStatuses(): ActiveSessionStatusById {
  const client = useAgentClient()
  const statuses = useMemo(() => Atom.make((get) =>
    Option.match(Result.value(get(client.query(Sessions.StreamActiveSessionStatuses, {})).result), {
      onNone: () => EMPTY,
      onSome: toStatusById,
    })), [client])
  return useAtomValue(statuses)
}
