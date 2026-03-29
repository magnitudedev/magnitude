import type { ContentPart } from '../content'
import type { ObservedResult, TurnToolCall } from '../events'
import type {
  AgentAtom,
  PhaseCriteriaPayload,
  ResultEntry,
  TimelineAttachment,
  TimelineEntry,
} from './types'

export interface ComposeContextDeps {
  resolveAgentByForkId(
    forkId: string,
  ): { agentId: string; role: string; parentForkId: string | null } | null
}

export function toResultToolResults(args: {
  toolCalls: readonly TurnToolCall[]
  observedResults: readonly ObservedResult[]
}): ResultEntry {
  return {
    kind: 'tool_results',
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


export function toTimelineWorkflowPhase(args: {
  timestamp: number
  name?: string
  phase?: string
  text: string
}): TimelineEntry {
  return {
    kind: 'workflow_phase',
    ...args,
  }
}

export function toTimelinePhaseCriteria(args: {
  timestamp: number
  payload: PhaseCriteriaPayload
}): TimelineEntry {
  return {
    kind: 'phase_criteria',
    ...args,
  }
}

export function toTimelinePhaseVerdict(args: {
  timestamp: number
  passed: boolean
  verdictText: string
  workflowCompleted: boolean
}): TimelineEntry {
  return {
    kind: 'phase_verdict',
    ...args,
  }
}

export function toTimelineSkillStarted(args: {
  timestamp: number
  skillName: string
  firstPhase?: string
  prompt: string
}): TimelineEntry {
  return {
    kind: 'skill_started',
    ...args,
  }
}

export function toTimelineSkillCompleted(args: {
  timestamp: number
  skillName: string
}): TimelineEntry {
  return {
    kind: 'skill_completed',
    ...args,
  }
}

export function toTimelineLifecycleHook(args: {
  timestamp: number
  agentId: string
  role: string
  hookType: 'spawn' | 'idle'
}): TimelineEntry {
  return {
    kind: 'lifecycle_hook',
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
