import { useEffect, useMemo, useRef, useState } from 'react'
import type { AgentStatusState, DisplayMessage, DisplayState } from '@magnitudedev/agent'
import type { TaskListItem, WorkerExecutionSnapshot } from '../components/chat/types'
import { flattenTaskTree, type TaskGraphState } from '../utils/task-tree'

type AgentClientLike = {
  state: {
    taskGraph: {
      subscribe: (cb: (state: TaskGraphState) => void) => () => void
    }
    display: {
      subscribeFork: (forkId: string | null, cb: (state: DisplayState) => void) => () => void
    }
    agentStatus: {
      subscribe: (cb: (state: AgentStatusState) => void) => () => void
    }
  }
}

type UseTasksArgs = {
  client: AgentClientLike | null
}

function getLatestForkActivityMessage(messages: readonly DisplayMessage[], workerForkId: string) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i]
    if (message?.type === 'fork_activity' && message.forkId === workerForkId) return message
  }
  return null
}

export function deriveWorkerExecutionSnapshot(args: {
  task: TaskListItem
  fromDisplay: WorkerExecutionSnapshot | null
  agentStatusState: AgentStatusState | null
}): WorkerExecutionSnapshot | null {
  const { task, fromDisplay, agentStatusState } = args
  if (task.assignee.kind !== 'worker' || !task.workerForkId) return null

  const linkedAgentId = agentStatusState?.agentByForkId.get(task.workerForkId)
  const linkedAgent = linkedAgentId ? agentStatusState?.agents.get(linkedAgentId) : undefined
  const canResolveAgent = agentStatusState !== null
  const isKilled = canResolveAgent && (linkedAgentId === undefined || linkedAgent === undefined)

  if (isKilled) {
    return {
      state: 'killed',
      activeSince: fromDisplay?.activeSince ?? null,
      accumulatedActiveMs: fromDisplay?.accumulatedActiveMs ?? 0,
      completedAt: fromDisplay?.completedAt ?? null,
      resumeCount: fromDisplay?.resumeCount ?? 0,
    }
  }

  const state = linkedAgent
    ? (linkedAgent.status === 'working' || linkedAgent.status === 'starting' ? 'working' : 'idle')
    : (fromDisplay?.state ?? 'idle')

  return {
    state,
    activeSince: fromDisplay?.activeSince ?? null,
    accumulatedActiveMs: fromDisplay?.accumulatedActiveMs ?? 0,
    completedAt: fromDisplay?.completedAt ?? null,
    resumeCount: fromDisplay?.resumeCount ?? 0,
  }
}

export function useTasks({ client }: UseTasksArgs): TaskListItem[] {
  const [tasks, setTasks] = useState<TaskListItem[]>([])
  const [agentStatusState, setAgentStatusState] = useState<AgentStatusState | null>(null)
  const [forkActivityByForkId, setForkActivityByForkId] = useState<Record<string, WorkerExecutionSnapshot>>({})
  const unsubscribesRef = useRef<Map<string, () => void>>(new Map())

  useEffect(() => {
    if (!client) {
      setTasks([])
      return
    }

    return client.state.taskGraph.subscribe((state) => {
      setTasks(flattenTaskTree(state))
    })
  }, [client])

  useEffect(() => {
    if (!client) {
      setAgentStatusState(null)
      return
    }

    return client.state.agentStatus.subscribe((state) => {
      setAgentStatusState(state)
    })
  }, [client])

  useEffect(() => {
    const workerForkIds = tasks.map(task => task.workerForkId).filter((id): id is string => Boolean(id))
    const activeSubscriptionKeys = new Set<string>()

    for (const workerForkId of workerForkIds) {
      const linkedAgentId = agentStatusState?.agentByForkId.get(workerForkId)
      const linkedAgent = linkedAgentId ? agentStatusState?.agents.get(linkedAgentId) : undefined
      const sourceForkId = linkedAgent?.parentForkId ?? null
      const sourceKey = sourceForkId ?? '__root__'
      const subscriptionKey = `${workerForkId}::${sourceKey}`
      activeSubscriptionKeys.add(subscriptionKey)

      if (!client || unsubscribesRef.current.has(subscriptionKey)) continue

      const unsubscribe = client.state.display.subscribeFork(sourceForkId, (state) => {
        const latest = getLatestForkActivityMessage(state.messages, workerForkId)
        if (!latest) return
        setForkActivityByForkId(prev => ({
          ...prev,
          [workerForkId]: {
            state: latest.status === 'running' ? 'working' : 'idle',
            activeSince: latest.activeSince ?? null,
            accumulatedActiveMs: latest.accumulatedActiveMs ?? 0,
            completedAt: latest.completedAt ?? null,
            resumeCount: latest.resumeCount ?? 0,
          },
        }))
      })
      unsubscribesRef.current.set(subscriptionKey, unsubscribe)
    }

    for (const [key, unsubscribe] of unsubscribesRef.current.entries()) {
      if (activeSubscriptionKeys.has(key)) continue
      unsubscribe()
      unsubscribesRef.current.delete(key)
    }

    const activeWorkerForkIdSet = new Set(workerForkIds)
    setForkActivityByForkId(prev => {
      let changed = false
      const next: Record<string, WorkerExecutionSnapshot> = {}
      for (const [forkId, snapshot] of Object.entries(prev)) {
        if (!activeWorkerForkIdSet.has(forkId)) {
          changed = true
          continue
        }
        next[forkId] = snapshot
      }
      return changed ? next : prev
    })
  }, [tasks, client, agentStatusState])

  useEffect(() => {
    return () => {
      for (const unsubscribe of unsubscribesRef.current.values()) unsubscribe()
      unsubscribesRef.current.clear()
    }
  }, [])

  return useMemo(() => {
    return tasks.map((task) => ({
      ...task,
      workerExecution: deriveWorkerExecutionSnapshot({
        task,
        fromDisplay: task.workerForkId ? (forkActivityByForkId[task.workerForkId] ?? null) : null,
        agentStatusState,
      }),
    }))
  }, [tasks, forkActivityByForkId, agentStatusState])
}