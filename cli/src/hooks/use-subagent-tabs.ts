import { useEffect, useMemo, useRef, useState } from 'react'
import type { AgentStatusState, DisplayMessage, DisplayState, ForkActivityMessage } from '@magnitudedev/agent'
import type { SubagentTabItem } from '../components/chat/types'
import {
  formatSubagentToolSummaryLine,
  sumForkToolCounts,
  truncateSubagentTabText,
} from '../utils/subagent-tabs'
import { selectLatestLiveActivityForSubagentTab } from '../utils/live-activity'

type AgentClientLike = {
  state: {
    display: {
      subscribeFork: (forkId: string | null, cb: (state: DisplayState) => void) => () => void
    }
  }
}

type UseSubagentTabsArgs = {
  client: AgentClientLike | null
  rootDisplayMessages: readonly DisplayMessage[]
  agentStatusState: AgentStatusState | null
  dismissedIdleForkIds?: ReadonlySet<string>
}

type ForkMeta = {
  agentId: string
  name: string
  activeSince: number
  accumulatedActiveMs: number
  completedAt?: number
  resumeCount: number
  toolCount: number
  toolCounts: ForkActivityMessage['toolCounts']
  phase: 'active' | 'idle'
}

export function buildSubagentTabItem(args: {
  forkId: string
  meta: ForkMeta
  messages: readonly DisplayMessage[]
  pendingDirect: { pending: boolean; since: number | null } | undefined
}): SubagentTabItem {
  const { forkId, meta, messages, pendingDirect } = args
  const toolSummaryLine = truncateSubagentTabText(formatSubagentToolSummaryLine(meta.toolCounts))
  const isPendingDirect = pendingDirect?.pending === true

  const phase: SubagentTabItem['phase'] = isPendingDirect ? 'active' : meta.phase
  const activeSince = phase === 'active'
    ? (meta.phase === 'active' ? meta.activeSince : (pendingDirect?.since ?? meta.activeSince))
    : meta.activeSince
  const completedAt = phase === 'active' ? undefined : meta.completedAt
  const accumulatedActiveMs = meta.accumulatedActiveMs

  const statusLine = truncateSubagentTabText(
    isPendingDirect
      ? 'User sent a message...'
      : meta.phase === 'idle'
        ? 'Subagent is idle'
        : (selectLatestLiveActivityForSubagentTab(messages) ?? 'Running…'),
  )

  return {
    forkId,
    agentId: meta.agentId,
    name: meta.name,
    activeSince,
    accumulatedActiveMs,
    completedAt,
    resumeCount: meta.resumeCount,
    toolCount: meta.toolCount,
    toolSummaryLine,
    statusLine,
    phase,
  }
}

export function sortSubagentTabs(a: SubagentTabItem, b: SubagentTabItem): number {
  if (a.phase !== b.phase) return a.phase === 'active' ? -1 : 1

  if (a.phase === 'idle' && b.phase === 'idle') {
    const aCompletedAt = a.completedAt ?? Number.NEGATIVE_INFINITY
    const bCompletedAt = b.completedAt ?? Number.NEGATIVE_INFINITY
    if (aCompletedAt !== bCompletedAt) return bCompletedAt - aCompletedAt
  }

  return a.activeSince - b.activeSince
}

function getAgentByForkId(agentStatusState: AgentStatusState | null, forkId: string) {
  if (!agentStatusState) return undefined
  return Array.from(agentStatusState.agents.values()).find(agent => agent.forkId === forkId)
}

export function reconcileForkMeta(args: {
  prev: Record<string, ForkMeta>
  latestByFork: ReadonlyMap<string, ForkActivityMessage>
  agentStatusState: AgentStatusState | null
}): { next: Record<string, ForkMeta> } {
  const { prev, latestByFork, agentStatusState } = args
  const next: Record<string, ForkMeta> = {}

  for (const [forkId, activity] of latestByFork.entries()) {
    const previous = prev[forkId]
    const forkAgent = getAgentByForkId(agentStatusState, forkId)

    const phase: ForkMeta['phase'] = activity.status === 'running' ? 'active' : 'idle'
    const completedAt = phase === 'active'
      ? undefined
      : (activity.completedAt ?? previous?.completedAt)

    next[forkId] = {
      agentId: forkAgent?.agentId ?? previous?.agentId ?? forkId,
      name: activity.name,
      activeSince: activity.activeSince,
      accumulatedActiveMs: activity.accumulatedActiveMs,
      completedAt,
      resumeCount: activity.resumeCount ?? 0,
      toolCount: sumForkToolCounts(activity.toolCounts),
      toolCounts: activity.toolCounts,
      phase,
    }
  }

  return { next }
}

export function useSubagentTabs({
  client,
  rootDisplayMessages,
  agentStatusState,
  dismissedIdleForkIds,
}: UseSubagentTabsArgs): SubagentTabItem[] {
  const [forkMessages, setForkMessages] = useState<Record<string, readonly DisplayMessage[]>>({})
  const [forkPendingDirectUser, setForkPendingDirectUser] = useState<Record<string, { pending: boolean; since: number | null }>>({})
  const [forkMeta, setForkMeta] = useState<Record<string, ForkMeta>>({})
  const unsubscribesRef = useRef<Map<string, () => void>>(new Map())

  const latestByFork = useMemo(() => {
    const map = new Map<string, ForkActivityMessage>()
    for (const message of rootDisplayMessages) {
      if (message.type !== 'fork_activity') continue
      map.set(message.forkId, message)
    }
    return map
  }, [rootDisplayMessages])

  useEffect(() => {
    setForkMeta(prev => {
      const reconciled = reconcileForkMeta({ prev, latestByFork, agentStatusState })
      const prevKeys = Object.keys(prev)
      const nextKeys = Object.keys(reconciled.next)
      if (prevKeys.length === nextKeys.length) {
        let changed = false
        for (const key of nextKeys) {
          if (prev[key] !== reconciled.next[key]) {
            changed = true
            break
          }
        }
        if (!changed) return prev
      }
      return reconciled.next
    })

    const activeForkIds = new Set(latestByFork.keys())

    for (const [forkId, unsubscribe] of unsubscribesRef.current.entries()) {
      if (activeForkIds.has(forkId)) continue
      unsubscribe()
      unsubscribesRef.current.delete(forkId)

      setForkMessages(prev => {
        if (!(forkId in prev)) return prev
        const { [forkId]: _removed, ...rest } = prev
        return rest
      })
      setForkPendingDirectUser(prev => {
        if (!(forkId in prev)) return prev
        const { [forkId]: _removed, ...rest } = prev
        return rest
      })
    }

    for (const forkId of activeForkIds) {
      if (!client || unsubscribesRef.current.has(forkId)) continue
      const unsubscribe = client.state.display.subscribeFork(forkId, (state) => {
        setForkMessages(prev => ({ ...prev, [forkId]: state.messages }))
        const pendingUser = state.pendingInboundCommunications.filter((message) => message.source === 'user')
        setForkPendingDirectUser(prev => ({
          ...prev,
          [forkId]: {
            pending: pendingUser.length > 0,
            since: pendingUser.length > 0
              ? Math.min(...pendingUser.map(message => message.timestamp))
              : null,
          },
        }))
      })
      unsubscribesRef.current.set(forkId, unsubscribe)
    }
  }, [latestByFork, client, agentStatusState])

  useEffect(() => {
    return () => {
      for (const unsubscribe of unsubscribesRef.current.values()) unsubscribe()
      unsubscribesRef.current.clear()
    }
  }, [])

  return useMemo(() => {
    return Object.entries(forkMeta)
      .map(([forkId, meta]): SubagentTabItem => (
        buildSubagentTabItem({
          forkId,
          meta,
          messages: forkMessages[forkId] ?? [],
          pendingDirect: forkPendingDirectUser[forkId],
        })
      ))
      .filter((tab) => !(tab.phase === 'idle' && dismissedIdleForkIds?.has(tab.forkId)))
      .sort(sortSubagentTabs)
  }, [forkMeta, forkMessages, forkPendingDirectUser, dismissedIdleForkIds])
}