import { getTaskTypeDefinition, listTaskTypeDefinitions, type TaskTypeId } from './registry'

/**
 * Lightweight reference table for system prompt.
 */
export function renderTaskTypeReferenceTable(): string {
  const lines: string[] = []
  const defs = listTaskTypeDefinitions()
  const compositeDefs = defs.filter((def) => def.kind === 'composite')
  const leafDefs = defs.filter((def) => def.kind === 'leaf' || def.kind === 'generic')
  const userDefs = defs.filter((def) => def.kind === 'user')

  lines.push('### Composite tasks')
  lines.push('Coordinator tasks that contain child tasks; not directly executed by workers.')
  for (const def of compositeDefs) {
    lines.push(`- **${def.label}** (\`${def.id}\`) — ${def.description}`)
  }

  lines.push('')
  lines.push('### Leaf tasks')
  lines.push('Directly executable tasks that are assigned to a worker role.')
  for (const def of leafDefs) {
    lines.push(`- **${def.label}** (\`${def.id}\`) — ${def.description}`)
    lines.push(`  Assignable to: ${def.allowedAssignees.join(', ')}`)
  }

  if (userDefs.length > 0) {
    lines.push('')
    lines.push('### User tasks')
    lines.push('Tasks that require a decision or approval from the user.')
    for (const def of userDefs) {
      lines.push(`- **${def.label}** (\`${def.id}\`) — ${def.description}`)
    }
  }

  return lines.join('\n')
}

export function formatTaskTypeGuidanceForTool(taskType: TaskTypeId): string {
  const def = getTaskTypeDefinition(taskType)

  const leadGuidance = def.leadGuidance.trim()
  const criteria = def.criteria.trim()

  if (!leadGuidance && !criteria) {
    return `No specific guidance for task type "${taskType}".`
  }

  if (!criteria) return leadGuidance
  if (!leadGuidance) return criteria

  return `${leadGuidance}\n\n${criteria}`
}

/**
 * task_type_hook reminder formatter — preserves task ID context and returns
 * full per-type lead guidance + criteria.
 */
export function formatTaskTypeReminder(taskIds: readonly string[], taskType: TaskTypeId): string {
  const idList = taskIds.length === 1 ? `Task ${taskIds[0]}` : `Tasks ${taskIds.join(', ')}`
  const guidance = formatTaskTypeGuidanceForTool(taskType)

  return `${idList} (type: ${taskType}):\n${guidance}`
}
