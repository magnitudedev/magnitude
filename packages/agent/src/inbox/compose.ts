import type { ContentPart } from '../content'
import type {
  AgentAtom,
  ResultEntry,
  TimelineAttachment,
  TimelineEntry,
  TurnResultItem,
} from './types'

export interface ComposeContextDeps {
  resolveAgentByForkId(
    forkId: string,
  ): { agentId: string; role: string; parentForkId: string | null } | null
}

export function toResultTurnResults(args: {
  items: readonly TurnResultItem[]
}): ResultEntry {
  return {
    kind: 'turn_results',
    ...args,
  }
}

export function toResultInterrupted(): ResultEntry {
  return {
    kind: 'interrupted',
  }
}

export function toResultError(args: { message: string }): ResultEntry {
  return {
    kind: 'error',
    ...args,
  }
}

export function toResultNoop(): ResultEntry {
  return {
    kind: 'noop',
  }
}

export function toTimelineUserMessage(args: {
  timestamp: number
  text: string
  attachments: readonly TimelineAttachment[]
}): TimelineEntry {
  return {
    kind: 'user_message',
    ...args,
  }
}

export function toTimelineParentMessage(args: {
  timestamp: number
  text: string
}): TimelineEntry {
  return {
    kind: 'parent_message',
    ...args,
  }
}

export function toTimelineUserBashCommand(args: {
  timestamp: number
  command: string
  cwd: string
  exitCode: number
  stdout: string
  stderr: string
}): TimelineEntry {
  return {
    kind: 'user_bash_command',
    ...args,
  }
}

export function toTimelineUserToAgent(args: {
  timestamp: number
  agentId: string
  text: string
}): TimelineEntry {
  return {
    kind: 'user_to_agent',
    ...args,
  }
}

export function toTimelineAgentBlock(args: {
  timestamp: number
  firstAtomTimestamp: number
  lastAtomTimestamp: number
  agentId: string
  role: string
  atoms: readonly AgentAtom[]
}): TimelineEntry {
  return {
    kind: 'agent_block',
    ...args,
  }
}

export function toTimelineSubagentUserKilled(args: {
  timestamp: number
  agentId: string
  agentType: string
}): TimelineEntry {
  return {
    kind: 'subagent_user_killed',
    ...args,
  }
}

export function toTimelineUserPresence(args: {
  timestamp: number
  text: string
  confirmed: boolean
}): TimelineEntry {
  return {
    kind: 'user_presence',
    ...args,
  }
}


export function toTimelineLifecycleHook(args: {
  timestamp: number
  agentId: string
  role: string
  hookType: 'spawn' | 'idle'
  taskId?: string
  taskTitle?: string
}): TimelineEntry {
  return {
    kind: 'lifecycle_hook',
    ...args,
  }
}

export function toTimelineTaskTypeHook(args: {
  timestamp: number
  taskId: string
  taskType: string
  title: string
}): TimelineEntry {
  return {
    kind: 'task_start_hook',
    ...args,
  }
}

export function toTimelineTaskIdleHook(args: {
  timestamp: number
  taskId: string
  taskType: string
  title: string
  agentId: string
}): TimelineEntry {
  return {
    kind: 'task_idle_hook',
    ...args,
  }
}

export function toTimelineTaskCompleteHook(args: {
  timestamp: number
  taskId: string
  taskType: string
  title: string
  skillPath: string
}): TimelineEntry {
  return {
    kind: 'task_complete_hook',
    ...args,
  }
}

export function toTimelineTaskTreeDirty(args: {
  timestamp: number
  taskId: string
}): TimelineEntry {
  return {
    kind: 'task_tree_dirty',
    ...args,
  }
}

export function toTimelineTaskTreeView(args: {
  timestamp: number
  renderedTree: string
}): TimelineEntry {
  return {
    kind: 'task_tree_view',
    ...args,
  }
}

export function toTimelineTaskUpdate(args: {
  timestamp: number
  action: 'created' | 'cancelled' | 'completed' | 'status_changed'
  taskId: string
  title?: string
  taskType?: string
  previousStatus?: string
  nextStatus?: string
  cancelledCount?: number
}): TimelineEntry {
  return {
    kind: 'task_update',
    ...args,
  }
}

export function toTimelineObservation(args: {
  timestamp: number
  parts: readonly ContentPart[]
}): TimelineEntry {
  return {
    kind: 'observation',
    ...args,
  }
}
