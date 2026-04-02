import { listTaskTypeDefinitions, type TaskTypeId } from './registry'

const HIGHER_ORDER_TASK_TYPES = new Set<TaskTypeId>(['feature', 'bug', 'refactor'])
const STRATEGY_ADHERENCE_REINFORCEMENT =
  '**Strategy adherence is mandatory.** The user EXPECTS this process to be followed. Failure to follow this workflow is a violation of user trust.\n\nEstablish the prescribed task decomposition immediately. If you have already started tasks that correspond to steps in this strategy, move them under the appropriate structure.'

/**
 * Lightweight reference table for system prompt.
 * Strategy guidance comes via inbox hooks on task creation.
 */
export function renderTaskTypeReferenceTable(): string {
  const lines: string[] = []
  for (const def of listTaskTypeDefinitions()) {
    lines.push(`- **${def.label}** (\`${def.id}\`) — ${def.description}`)
    lines.push(`  Assignees: ${def.allowedAssignees.join(', ')}`)
  }
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
  if (HIGHER_ORDER_TASK_TYPES.has(taskType)) {
    lines.push(STRATEGY_ADHERENCE_REINFORCEMENT)
  }
  lines.push(def.strategy)
  return lines.join('\n')
}
