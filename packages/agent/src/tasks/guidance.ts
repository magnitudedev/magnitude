import { getTaskTypeDefinition, isTaskTypeKind, listTaskTypeDefinitions, type TaskTypeId } from './registry'
const STRATEGY_ADHERENCE_REINFORCEMENT =
  '**Strategy adherence is mandatory.** The user EXPECTS this process to be followed. Failure to follow this workflow is a violation of user trust.\n\nEstablish the prescribed task decomposition immediately. If you have already started tasks that correspond to steps in this strategy, move them under the appropriate structure.'

/**
 * Lightweight reference table for system prompt.
 */
export function renderTaskTypeReferenceTable(): string {
  const lines: string[] = []
  for (const def of listTaskTypeDefinitions()) {
    lines.push(`- **${def.label}** (\`${def.id}\`) — ${def.description}`)
    const assignees = def.allowedAssignees.length > 0
      ? def.allowedAssignees.join(', ')
      : 'none (coordinator task; create/assign child tasks)'
    lines.push(`  Assignees: ${assignees}`)
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
 * task_type_hook reminder formatter — reinforcement only, preserves task ID context.
 */
export function formatTaskTypeReminder(taskIds: readonly string[], taskType: TaskTypeId): string {
  if (!isTaskTypeKind(taskType, 'composite')) return ''

  const idList = taskIds.length === 1 ? `Task ${taskIds[0]}` : `Tasks ${taskIds.join(', ')}`
  return `${idList} (type: ${taskType}):\n${STRATEGY_ADHERENCE_REINFORCEMENT}`
}
