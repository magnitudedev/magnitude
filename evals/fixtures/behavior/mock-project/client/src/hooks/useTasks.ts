import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'

export type Task = {
  id: string
  title: string
  description?: string | null
  status: 'todo' | 'in-progress' | 'done'
}

export function useTasks(projectId: string | null, token: string | null) {
  const [tasks, setTasks] = useState<Task[]>([])

  useEffect(() => {
    if (!projectId || !token) return
    apiClient<Task[]>(`/api/projects/${projectId}/tasks`, {}, token).then(setTasks).catch(() => setTasks([]))
  }, [projectId, token])

  return { tasks, setTasks }
}