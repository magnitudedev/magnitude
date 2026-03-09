import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'

export type Project = { id: string; name: string; description?: string | null }

export function useProjects(token: string | null) {
  const [projects, setProjects] = useState<Project[]>([])

  useEffect(() => {
    if (!token) return
    apiClient<Project[]>('/api/projects', {}, token).then(setProjects).catch(() => setProjects([]))
  }, [token])

  return { projects, setProjects }
}