import { useEffect, useMemo, useRef, useState } from 'react'
import type { DisplayState } from '@magnitudedev/agent'
import type { TaskListItem } from '../components/chat/types'
import { flattenTaskTree, type TaskGraphState } from '../utils/task-tree'

type AgentClientLike = {
  state: {
    taskGraph: {
      subscribe: (cb: (state: TaskGraphState) => void) => () => void
    }
    display: {
      subscribeFork: (forkId: string | null, cb: (state: DisplayState) => void) => () => void
    }
  }
}

type UseTasksArgs = {
  client: AgentClientLike | null
}

export function useTasks({ client }: UseTasksArgs): TaskListItem[] {
  const [tasks, setTasks] = useState<TaskListItem[]>([])
  const [forkPendingDirectUser, setForkPendingDirectUser] = useState<Record<string, { pending: boolean; since: number | null }>>({})
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
    const activeForkIds = new Set(tasks.map(task => task.workerForkId).filter((id): id is string => Boolean(id)))

    for (const [forkId, unsubscribe] of unsubscribesRef.current.entries()) {
      if (activeForkIds.has(forkId)) continue
      unsubscribe()
      unsubscribesRef.current.delete(forkId)

      setForkPendingDirectUser(prev => {
        if (!(forkId in prev)) return prev
        const { [forkId]: _removed, ...rest } = prev
        return rest
      })
    }

    for (const forkId of activeForkIds) {
      if (!client || unsubscribesRef.current.has(forkId)) continue
      const unsubscribe = client.state.display.subscribeFork(forkId, (state) => {
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
  }, [tasks, client])

  useEffect(() => {
    return () => {
      for (const unsubscribe of unsubscribesRef.current.values()) unsubscribe()
      unsubscribesRef.current.clear()
    }
  }, [])

  return useMemo(() => tasks.map((task) => {
    if (task.status === 'completed') return task
    if (!task.workerForkId) return task
    const pending = forkPendingDirectUser[task.workerForkId]?.pending === true
    return pending ? { ...task, status: 'working' } : task
  }), [tasks, forkPendingDirectUser])
}