import { listTaskTypeDefinitions, type TaskTypeId } from './registry'

const HIGHER_ORDER_TASK_TYPES = new Set<TaskTypeId>(['feature', 'bug', 'refactor'])
const STRATEGY_ADHERENCE_REINFORCEMENT =
  '**Strategy adherence is mandatory.** The user EXPECTS this process to be followed. Failure to follow this workflow is a violation of user trust.\n\nEstablish the prescribed task decomposition immediately. If you have already started tasks that correspond to steps in this strategy, move them under the appropriate structure.'

/**
 * Lightweight reference table for system prompt.
 */
export function renderTaskTypeReferenceTable(): string {
  const lines: string[] = []
  for (const def of listTaskTypeDefinitions()) {
    lines.push(`- **${def.label}** (\`${def.id}\`) — ${def.description}`)
    lines.push(`  Assignees: ${def.allowedAssignees.join(', ')}`)
  }
  return lines.join('\n')
}

export function formatTaskTypeGuidanceForTool(taskType: TaskTypeId): string {
  const def = listTaskTypeDefinitions().find((d) => d.id === taskType)
  if (!def) return `No specific guidance for task type "${taskType}".`

  const guidance = def.strategy.trim()
  if (!guidance) return `No specific guidance for task type "${taskType}".`

  return guidance
}

/**
 * task_type_hook reminder formatter — reinforcement only, preserves task ID context.
 */
export function formatTaskTypeReminder(taskIds: readonly string[], taskType: TaskTypeId): string {
  if (!HIGHER_ORDER_TASK_TYPES.has(taskType)) return ''

  const idList = taskIds.length === 1 ? `Task ${taskIds[0]}` : `Tasks ${taskIds.join(', ')}`
  return `${idList} (type: ${taskType}):\n${STRATEGY_ADHERENCE_REINFORCEMENT}`
}
