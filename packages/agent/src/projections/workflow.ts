import { Projection, Signal } from '@magnitudedev/event-core'
import { createWorkflowState, reduce, type WorkflowState } from '@magnitudedev/skills'
import type { AppEvent, PhaseCriteriaVerdict } from '../events'

export interface WorkflowCriteriaState {
  readonly workflowState: WorkflowState | null
  readonly expectedCount: number
  readonly criteria: Array<{
    index: number
    name: string
    type: 'shell' | 'agent' | 'user'
    status: 'pending' | 'running' | 'passed' | 'failed'
    reason?: string
  }>
  readonly shellCount: number
  readonly shellResolved: number
  readonly allShellPassed: boolean
  readonly resolved: boolean
  readonly skillName: string
  readonly phases: Array<{
    name: string
    status: 'pending' | 'active' | 'verifying' | 'completed'
  }>
}

export interface PhaseVerdictSignalEntry {
  readonly criteriaIndex: number
  readonly criteriaName: string
  readonly passed: boolean
  readonly reason: string
}

function toVerdicts(state: WorkflowCriteriaState): PhaseVerdictSignalEntry[] {
  return state.criteria
    .filter((c) => c.status === 'passed' || c.status === 'failed')
    .map((c) => ({
      criteriaIndex: c.index,
      criteriaName: c.name,
      passed: c.status === 'passed',
      reason: c.reason ?? (c.status === 'passed' ? 'passed' : 'failed'),
    }))
}

function applyVerdict(state: WorkflowCriteriaState, event: PhaseCriteriaVerdict): WorkflowCriteriaState {
  const criteria = state.criteria.map((item) => {
    if (item.index !== event.criteriaIndex) return item
    return {
      ...item,
      status: event.status,
      reason: 'reason' in event ? event.reason : item.reason,
    }
  })

  const shellResolved = criteria.filter((c) => c.type === 'shell' && (c.status === 'passed' || c.status === 'failed')).length
  const allShellPassed = state.shellCount === 0
    ? true
    : shellResolved === state.shellCount && criteria.filter((c) => c.type === 'shell').every((c) => c.status === 'passed')

  return {
    ...state,
    criteria,
    shellResolved,
    allShellPassed,
  }
}

const initialFork: WorkflowCriteriaState = {
  workflowState: null,
  expectedCount: 0,
  criteria: [],
  shellCount: 0,
  shellResolved: 0,
  allShellPassed: false,
  resolved: false,
  skillName: '',
  phases: [],
}

export const WorkflowProjection = Projection.defineForked<AppEvent, WorkflowCriteriaState>()({
  name: 'Workflow',

  initialFork,

  signals: {
    shellCriteriaPassed: Signal.create<{ forkId: string | null }>('Workflow/shellCriteriaPassed'),
    phaseResolved: Signal.create<{ forkId: string | null; passed: boolean; verdicts: readonly PhaseVerdictSignalEntry[] }>('Workflow/phaseResolved'),
  },

  eventHandlers: {
    skill_started: ({ event, fork }) => ({
      ...fork,
      workflowState: createWorkflowState(event.skill),
      skillName: event.skill.name,
      phases: event.skill.phases.map((phase, index) => ({
        name: phase.name,
        status: index === 0 ? 'active' : 'pending',
      })),
    }),

    phase_submitted: ({ event, fork }) => {
      if (!fork.workflowState) return fork
      return {
        ...fork,
        workflowState: reduce(fork.workflowState, { type: 'submit', fields: event.fields }),
      }
    },

    phase_criteria_started: ({ event, fork }) => {
      const shellCount = event.criteria.filter((c) => c.type === 'shell').length
      return {
        ...fork,
        expectedCount: event.criteria.length,
        criteria: event.criteria.map((c) => ({
          index: c.index,
          name: c.name,
          type: c.type,
          status: c.type === 'shell' ? 'running' : 'pending',
        })),
        shellCount,
        shellResolved: 0,
        allShellPassed: shellCount === 0,
        resolved: false,
        phases: fork.phases.map((phase) => phase.status === 'active' ? { ...phase, status: 'verifying' } : phase),
      }
    },

    phase_verdict: ({ event, fork }) => {
      let nextWorkflowState = fork.workflowState

      if (nextWorkflowState) {
        nextWorkflowState = event.passed
          ? reduce(nextWorkflowState, { type: 'advance' })
          : reduce(nextWorkflowState, { type: 'criteria-failed', results: [] })
      }

      if (!event.workflowCompleted) {
        const activeIndex = fork.phases.findIndex((phase) => phase.status === 'active' || phase.status === 'verifying')
        if (activeIndex === -1) {
          return {
            ...fork,
            workflowState: nextWorkflowState,
          }
        }
        return {
          ...fork,
          workflowState: nextWorkflowState,
          phases: fork.phases.map((phase, index) => {
            if (index < activeIndex) return { ...phase, status: 'completed' as const }
            if (index === activeIndex) return event.passed ? { ...phase, status: 'completed' as const } : { ...phase, status: 'active' as const }
            if (index === activeIndex + 1 && event.passed) return { ...phase, status: 'active' as const }
            return phase
          }),
        }
      }

      return {
        ...fork,
        workflowState: nextWorkflowState,
        phases: fork.phases.map((phase) => ({ ...phase, status: 'completed' as const })),
      }
    },

    skill_completed: ({ fork }) => {
      return {
        ...fork,
        workflowState: null,
        skillName: '',
        phases: [],
        criteria: [],
      }
    },
  },

  globalEventHandlers: {
    phase_criteria_verdict: ({ event, state, emit }) => {
      const targetForkId = event.parentForkId
      const fork = state.forks.get(targetForkId)
      if (!fork || fork.resolved) return state

      const next = applyVerdict(fork, event)
      const failed = next.criteria.some((c) => c.status === 'failed')
      if (failed) {
        const resolved = { ...next, resolved: true }
        emit.phaseResolved({
          forkId: targetForkId,
          passed: false,
          verdicts: toVerdicts(resolved),
        })
        return { ...state, forks: new Map(state.forks).set(targetForkId, resolved) }
      }

      if (!fork.allShellPassed && next.allShellPassed) {
        emit.shellCriteriaPassed({ forkId: targetForkId })
      }

      const resolvedCount = next.criteria.filter((c) => c.status === 'passed' || c.status === 'failed').length
      if (resolvedCount === next.expectedCount && next.expectedCount > 0) {
        const resolved = { ...next, resolved: true }
        emit.phaseResolved({
          forkId: targetForkId,
          passed: true,
          verdicts: toVerdicts(resolved),
        })
        return { ...state, forks: new Map(state.forks).set(targetForkId, resolved) }
      }

      return { ...state, forks: new Map(state.forks).set(targetForkId, next) }
    },
  },
})
