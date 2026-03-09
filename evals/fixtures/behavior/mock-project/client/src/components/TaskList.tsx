import type { Task } from '../hooks/useTasks'
import { TaskCard } from './TaskCard'

export function TaskList({ tasks }: { tasks: Task[] }) {
  return (
    <section>
      <h2>Tasks</h2>
      {tasks.map((task) => (
        <TaskCard key={task.id} task={task} />
      ))}
    </section>
  )
}