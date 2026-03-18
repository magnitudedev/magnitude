import { useEffect, useMemo, useRef, useState } from 'react'
import type { AgentStatusState, DisplayMessage, DisplayState, ForkActivityMessage } from '@magnitudedev/agent'
import type { SubagentTabItem } from '../components/chat/types'
import { deriveSubagentStatusLine, sumForkToolCounts, truncateSubagentTabText } from '../utils/subagent-tabs'

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
  exitMs?: number
}

type ForkMeta = {
  agentId: string
  name: string
  startedAt: number
  toolCount: number
  phase: 'active' | 'exiting'
}

const DEFAULT_EXIT_MS = 1000

export function useSubagentTabs({
  client,
  rootDisplayMessages,
  agentStatusState,
  exitMs = DEFAULT_EXIT_MS,
}: UseSubagentTabsArgs): SubagentTabItem[] {
  const [forkMessages, setForkMessages] = useState<Record<string, readonly DisplayMessage[]>>({})
  const [forkMeta, setForkMeta] = useState<Record<string, ForkMeta>>({})
  const unsubscribesRef = useRef<Map<string, () => void>>(new Map())
  const exitTimersRef = useRef<Map<string, NodeJS.Timeout>>(new Map())

  const latestByFork = useMemo(() => {
    const map = new Map<string, ForkActivityMessage>()
    for (const message of rootDisplayMessages) {
      if (message.type !== 'fork_activity') continue
      map.set(message.forkId, message)
    }
    return map
  }, [rootDisplayMessages])

  useEffect(() => {
    const activeForkIds = new Set<string>()

    for (const [forkId, activity] of latestByFork.entries()) {
      if (activity.status !== 'running') continue
      if (agentStatusState) {
        const forkAgent = Array.from(agentStatusState.agents.values()).find(a => a.forkId === forkId)
        if (!forkAgent || forkAgent.status !== 'working') continue
      }
      activeForkIds.add(forkId)
      const forkAgent = agentStatusState
        ? Array.from(agentStatusState.agents.values()).find(a => a.forkId === forkId)
        : undefined
      setForkMeta(prev => {
        const existing = prev[forkId]
        const nextMeta: ForkMeta = {
          agentId: forkAgent?.agentId ?? forkId,
          name: activity.name,
          startedAt: activity.startedAt,
          toolCount: sumForkToolCounts(activity.toolCounts),
          phase: 'active',
        }
        if (
          existing
          && existing.agentId === nextMeta.agentId
          && existing.name === nextMeta.name
          && existing.startedAt === nextMeta.startedAt
          && existing.toolCount === nextMeta.toolCount
          && existing.phase === nextMeta.phase
        ) {
          return prev
        }
        return {
          ...prev,
          [forkId]: nextMeta,
        }
      })
      const existingTimer = exitTimersRef.current.get(forkId)
      if (existingTimer) {
        clearTimeout(existingTimer)
        exitTimersRef.current.delete(forkId)
      }

      if (!client || unsubscribesRef.current.has(forkId)) continue
      const unsubscribe = client.state.display.subscribeFork(forkId, (state) => {
        setForkMessages(prev => ({ ...prev, [forkId]: state.messages }))
      })
      unsubscribesRef.current.set(forkId, unsubscribe)
    }

    for (const [forkId, meta] of Object.entries(forkMeta)) {
      if (meta.phase !== 'active') continue
      if (activeForkIds.has(forkId)) continue

      setForkMeta(prev => {
        const next = prev[forkId]
        if (!next || next.phase === 'exiting') return prev
        return {
          ...prev,
          [forkId]: { ...next, phase: 'exiting' },
        }
      })

      if (!exitTimersRef.current.has(forkId)) {
        const timeout = setTimeout(() => {
          const unsubscribe = unsubscribesRef.current.get(forkId)
          if (unsubscribe) {
            unsubscribe()
            unsubscribesRef.current.delete(forkId)
          }
          exitTimersRef.current.delete(forkId)
          setForkMeta(prev => {
            const next = { ...prev }
            delete next[forkId]
            return next
          })
          setForkMessages(prev => {
            const next = { ...prev }
            delete next[forkId]
            return next
          })
        }, exitMs)
        exitTimersRef.current.set(forkId, timeout)
      }
    }
  }, [latestByFork, client, agentStatusState, forkMeta, exitMs])

  useEffect(() => {
    return () => {
      for (const unsubscribe of unsubscribesRef.current.values()) unsubscribe()
      unsubscribesRef.current.clear()
      for (const timeout of exitTimersRef.current.values()) clearTimeout(timeout)
      exitTimersRef.current.clear()
    }
  }, [])

  return useMemo(() => {
    return Object.entries(forkMeta)
      .map(([forkId, meta]): SubagentTabItem => {
        const statusLine = truncateSubagentTabText(deriveSubagentStatusLine(forkMessages[forkId] ?? []))
        return {
          forkId,
          agentId: meta.agentId,
          name: meta.name,
          startedAt: meta.startedAt,
          toolCount: meta.toolCount,
          statusLine,
          phase: meta.phase,
        }
      })
      .sort((a, b) => a.startedAt - b.startedAt)
  }, [forkMeta, forkMessages])
}