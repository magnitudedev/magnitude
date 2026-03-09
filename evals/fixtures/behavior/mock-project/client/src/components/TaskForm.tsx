import { useState } from 'react'

export function TaskForm({ onCreate }: { onCreate: (title: string, description: string) => void }) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        onCreate(title, description)
        setTitle('')
        setDescription('')
      }}
    >
      <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Task title" />
      <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Task description" />
      <button type="submit">Add Task</button>
    </form>
  )
}