import { useState } from 'react'
import { apiClient } from './api/client'
import { Header } from './components/Header'
import { ProjectList } from './components/ProjectList'
import { TaskForm } from './components/TaskForm'
import { TaskList } from './components/TaskList'
import { useAuth } from './hooks/useAuth'
import { useProjects } from './hooks/useProjects'
import { useTasks } from './hooks/useTasks'

export default function App() {
  const { token, login } = useAuth()
  const { projects } = useProjects(token)
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null)
  const { tasks, setTasks } = useTasks(activeProjectId, token)

  return (
    <main>
      <Header />
      {!token ? (
        <button onClick={() => login('admin@example.com', 'password123')}>Login Demo User</button>
      ) : (
        <>
          <ProjectList projects={projects} activeId={activeProjectId} onSelect={setActiveProjectId} />
          {activeProjectId && (
            <>
              <TaskForm
                onCreate={async (title, description) => {
                  if (!token) return
                  const created = await apiClient<any>(
                    `/api/projects/${activeProjectId}/tasks`,
                    {
                      method: 'POST',
                      body: JSON.stringify({ title, description }),
                    },
                    token,
                  )
                  setTasks((prev) => [...prev, created])
                }}
              />
              <TaskList tasks={tasks} />
            </>
          )}
        </>
      )}
    </main>
  )
}