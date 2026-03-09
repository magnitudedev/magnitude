import type { Project } from '../hooks/useProjects'

export function ProjectList({
  projects,
  activeId,
  onSelect,
}: {
  projects: Project[]
  activeId: string | null
  onSelect: (id: string) => void
}) {
  return (
    <aside>
      <h2>Projects</h2>
      {projects.map((p) => (
        <button key={p.id} onClick={() => onSelect(p.id)} style={{ fontWeight: activeId === p.id ? 'bold' : 'normal' }}>
          {p.name}
        </button>
      ))}
    </aside>
  )
}