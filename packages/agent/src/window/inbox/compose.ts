import type { UserPart } from '@magnitudedev/ai'
import { Option } from 'effect'
import type { ObserverJustification } from '../../observer'
import type {
  AgentAtom,
  TimelineAttachment,
  TimelineEntry,
} from './types'

export function toTimelineTurnStart(args: {
  timestamp: number
  turnId: string
}): TimelineEntry {
  return {
    kind: 'turn_start',
    ...args,
  }
}

export function toTimelineTurnEnd(args: {
  timestamp: number
  turnId: string
}): TimelineEntry {
  return {
    kind: 'turn_end',
    ...args,
  }
}

export interface ComposeContextDeps {
  resolveAgentByForkId(
    forkId: string,
  ): { agentId: string; role: string; parentForkId: string | null } | null
}

export function toTimelineUserMessage(args: {
  timestamp: number
  text: string
  attachments: readonly TimelineAttachment[]
  synthetic: Option.Option<boolean>
}): TimelineEntry {
  return {
    kind: 'user_message',
    ...args,
  }
}

export function toTimelineCoordinatorMessage(args: {
  timestamp: number
  text: string
}): TimelineEntry {
  return {
    kind: 'coordinator_message',
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
  status: string
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
    kind: 'worker_user_killed',
    ...args,
  }
}

export function toTimelineLifecycleHook(args: {
  timestamp: number
  agentId: string
  role: string
  hookType: 'spawn' | 'idle'
  taskId: Option.Option<string>
  taskTitle: Option.Option<string>
}): TimelineEntry {
  return {
    kind: 'lifecycle_hook',
    ...args,
  }
}

export function toTimelineTaskTypeHook(args: {
  timestamp: number
  taskId: string
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
  title: string
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
  title: Option.Option<string>
  previousStatus: Option.Option<string>
  nextStatus: Option.Option<string>
  cancelledCount: Option.Option<number>
}): TimelineEntry {
  return {
    kind: 'task_update',
    ...args,
  }
}

export function toTimelineTaskReassigned(args: {
  timestamp: number
  oldTaskId: string
  newTaskId: string
}): TimelineEntry {
  return {
    kind: 'task_reassigned',
    ...args,
    text: '',
  }
}

export function toTimelineObservation(args: {
  timestamp: number
  parts: readonly UserPart[]
}): TimelineEntry {
  return {
    kind: 'observation',
    ...args,
  }
}

export function toTimelineDetachedProcessExited(args: {
  timestamp: number
  pid: number
  command: string
  exitCode: number
  stdoutPath: string
  stderrPath: string
}): TimelineEntry {
  return {
    kind: 'detached_process_exited',
    ...args,
  }
}

export function toTimelineEscalation(args: {
  timestamp: number
  observedForkId: string | null
  observedTurnId: string
  justification: ObserverJustification | null
  coalesceKey: Option.Option<string>
}): TimelineEntry {
  return {
    kind: 'escalation',
    ...args,
  }
}

export function toTimelineBackgroundProcesses(args: {
  timestamp: number
  processes: readonly {
    readonly pid: number
    readonly command: string
    readonly elapsedMs: number
    readonly cpuPercent: number | null
    readonly rssBytes: number | null
    readonly ownerAgentId: Option.Option<string>
  }[]
}): TimelineEntry {
  return {
    kind: 'background_processes',
    ...args,
  }
}
