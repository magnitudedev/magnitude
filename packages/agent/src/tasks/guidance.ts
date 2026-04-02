import { listTaskTypeDefinitions, type TaskTypeId } from './registry'

/**
 * Lightweight reference table for system prompt — types + allowed assignees only.
 * Strategy guidance comes via inbox hooks on task creation.
 */
export function renderTaskTypeReferenceTable(): string {
  const lines: string[] = []
  lines.push('<task_types>')
  for (const def of listTaskTypeDefinitions()) {
    lines.push(`  <type id="${def.id}" label="${def.label}" assignees="${def.allowedAssignees.join(', ')}" />`)
  }
  lines.push('</task_types>')
  return lines.join('\n')
}

/**
 * Task creation reminder formatter — called by inbox system when task_type_hook
 * timeline entries are rendered. Receives consolidated taskIds grouped by type.
 */
export function formatTaskTypeReminder(taskIds: readonly string[], taskType: TaskTypeId): string {
  const def = listTaskTypeDefinitions().find(d => d.id === taskType)
  if (!def) return `Tasks ${taskIds.join(', ')} created (unknown type: ${taskType}).`

  const idList = taskIds.length === 1 ? `Task ${taskIds[0]}` : `Tasks ${taskIds.join(', ')}`
  const lines: string[] = []
  lines.push(`${idList} (type: ${def.label}):`)
  lines.push(def.strategy)
  return lines.join('\n')
}
