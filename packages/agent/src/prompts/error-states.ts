import { TURN_CONTROL_IDLE } from '@magnitudedev/xml-act'

export const UNCLOSED_THINK_REMINDER = 'Your response had an unclosed thinking block. Be careful to use structural tags correctly and avoid referencing them in your thinking or prose.'

export const UNCLOSED_TASK_REMINDER = 'Your response had an unclosed task block. Be careful to use structural tags correctly and avoid referencing them in your thinking or prose.'

export const ONESHOT_LIVENESS_REMINDER = 'You idled but no subagents are active and there is no user to respond. Continue working toward the task or call <finish/> when complete.'

export function formatSpawnNoMessageReminder(taskId: string, taskTitle: string, role: string): string {
  return `Worker \`${role}\` was spawned on task ${taskId} ("${taskTitle}") but has no instructions yet — it is idle until it receives a message. Send a \`<message to="${taskId}">\` to assign it work.`
}

export function formatNonexistentAgentError(destList: string): string {
  return `Message sent to nonexistent agent ID(s): ${destList}. The message was not delivered. Check the agent ID and ensure the agent has been created and is still active.`
}

export function formatTaskOutsideSubtreeError(taskId: string, attemptedParent: string, assignedTaskId: string): string {
  return `Task creation rejected: workers can only create subtasks under assigned task "${assignedTaskId}". Task "${taskId}" with parent "${attemptedParent}" is outside your subtree.`
}

// Task type validation removed - types are now optional and not validated against skills

export function formatTaskNotFoundError(taskId: string): string {
  return `Task operation rejected: task "${taskId}" does not exist.`
}

export function formatTaskParentNotFoundError(taskId: string, parentId: string): string {
  return `Task operation rejected: parent task "${parentId}" was not found for task "${taskId}".`
}

export function formatDuplicateTaskIdError(taskId: string): string {
  return `Task creation rejected: task "${taskId}" already exists. Task IDs must be unique.`
}

export function formatTaskCompletionBlockedError(taskId: string): string {
  return `Task update rejected: cannot mark task "${taskId}" as completed while child tasks are incomplete.`
}

export function formatInvalidAssigneeError(taskId: string, assignee: string): string {
  return `Task assignment rejected: assignee "${assignee}" is invalid for task "${taskId}".`
}

export function formatMissingAssignmentMessageError(taskId: string): string {
  return `Task assignment rejected: missing assignment message for task "${taskId}".`
}

export function formatMissingAssignmentRoleError(taskId: string): string {
  return `Task assignment rejected: role is required when task "${taskId}" has no active worker.`
}

export const EMPTY_RESPONSE_ERROR = `Your response was empty. You must respond with lenses/messages/tool calls. Use \`${TURN_CONTROL_IDLE}\` if done taking turns.`
