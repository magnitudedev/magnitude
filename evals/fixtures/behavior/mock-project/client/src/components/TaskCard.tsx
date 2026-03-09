import type { Task } from '../hooks/useTasks'

export function TaskCard({ task }: { task: Task }) {
  return (
    <article>
      <h4>{task.title}</h4>
      <p>{task.description}</p>
      <small>{task.status}</small>
    </article>
  )
}