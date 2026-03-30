import { Data } from 'effect'
import { Projection, Signal, FSM } from '@magnitudedev/event-core'
import { createWorkflowState, reduce, type WorkflowState } from '@magnitudedev/skills'
import type { AppEvent, PhaseCriteriaVerdict } from '../events'

const { defineFSM } = FSM

export class CriterionPending extends Data.TaggedClass('pending')<{}> {}
export class CriterionRunning extends Data.TaggedClass('running')<{}> {}
export class CriterionPassed extends Data.TaggedClass('passed')<{}> {}
export class CriterionFailed extends Data.TaggedClass('failed')<{ readonly reason: string }> {}

export const CriterionLifecycle = defineFSM(
  { pending: CriterionPending, running: CriterionRunning, passed: CriterionPassed, failed: CriterionFailed },
  {
    pending: ['running', 'passed', 'failed'],
    running: ['passed', 'failed', 'pending'],
    passed: [],
    failed: [],
  }
)

export type CriterionLifecycleState =
  | CriterionPending
  | CriterionRunning
  | CriterionPassed
  | CriterionFailed

export interface WorkflowCriterion {
  readonly index: number
  readonly name: string
  readonly type: 'shell' | 'agent' | 'user'
  readonly lifecycle: CriterionLifecycleState
}

export interface WorkflowCriteriaState {
  readonly workflowState: WorkflowState | null
  readonly expectedCount: number
  readonly criteria: WorkflowCriterion[]
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

function hasPendingVerdict(criteria: readonly WorkflowCriterion[]): boolean {
  return criteria.some((c) => c.lifecycle._tag === 'pending' || c.lifecycle._tag === 'running')
}

function toVerdicts(state: WorkflowCriteriaState): PhaseVerdictSignalEntry[] {
  return state.criteria
    .filter((c) => c.lifecycle._tag === 'passed' || c.lifecycle._tag === 'failed')
    .map((c) => ({
      criteriaIndex: c.index,
      criteriaName: c.name,
      passed: c.lifecycle._tag === 'passed',
      reason: c.lifecycle._tag === 'failed' ? c.lifecycle.reason : 'passed',
    }))
}

function applyVerdict(state: WorkflowCriteriaState, event: PhaseCriteriaVerdict): WorkflowCriteriaState {
  const criteria = state.criteria.map((item) => {
    if (item.index !== event.criteriaIndex) return item
    if (event.status === 'passed' && item.lifecycle._tag !== 'passed') {
      return { ...item, lifecycle: CriterionLifecycle.transition(item.lifecycle, 'passed', {}) }
    }
    if (event.status === 'failed' && item.lifecycle._tag !== 'failed') {
      const reason = 'reason' in event ? event.reason : 'failed'
      return { ...item, lifecycle: CriterionLifecycle.transition(item.lifecycle, 'failed', { reason }) }
    }
    return item
  })

  const shellResolved = criteria.filter((c) =>
    c.type === 'shell' && (c.lifecycle._tag === 'passed' || c.lifecycle._tag === 'failed')
  ).length

  const allShellPassed = state.shellCount === 0
    ? true
    : shellResolved === state.shellCount && criteria.filter((c) => c.type === 'shell').every((c) => c.lifecycle._tag === 'passed')

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
    verdictPendingChanged: Signal.create<{ forkId: string | null; pending: boolean }>('Workflow/verdictPendingChanged'),
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

    phase_criteria_started: ({ event, fork, emit }) => {
      const previousPending = hasPendingVerdict(fork.criteria)
      const shellCount = event.criteria.filter((c) => c.type === 'shell').length
      const criteria: WorkflowCriterion[] = event.criteria.map((c) => ({
        index: c.index,
        name: c.name,
        type: c.type,
        lifecycle: c.type === 'shell' ? new CriterionRunning() : new CriterionPending(),
      }))
      const nextPending = hasPendingVerdict(criteria)

      if (previousPending !== nextPending) {
        emit.verdictPendingChanged({ forkId: event.forkId, pending: nextPending })
      }

      return {
        ...fork,
        expectedCount: event.criteria.length,
        criteria,
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
            if (index < activeIndex) return { ...phase, status: 'completed' }
            if (index === activeIndex) return event.passed ? { ...phase, status: 'completed' } : { ...phase, status: 'active' }
            if (index === activeIndex + 1 && event.passed) return { ...phase, status: 'active' }
            return phase
          }),
        }
      }

      return {
        ...fork,
        workflowState: nextWorkflowState,
        phases: fork.phases.map((phase) => ({ ...phase, status: 'completed' })),
      }
    },

    skill_completed: ({ event, fork, emit }) => {
      const previousPending = hasPendingVerdict(fork.criteria)
      if (previousPending) {
        emit.verdictPendingChanged({ forkId: event.forkId, pending: false })
      }
      return {
        ...fork,
        workflowState: null,
        skillName: '',
        phases: [],
        criteria: [],
      }
    },

    interrupt: ({ event, fork, emit }) => {
      const previousPending = hasPendingVerdict(fork.criteria)
      const criteria = fork.criteria.map((criterion) =>
        criterion.lifecycle._tag === 'running'
          ? { ...criterion, lifecycle: CriterionLifecycle.transition(criterion.lifecycle, 'pending', {}) }
          : criterion
      )
      const nextPending = hasPendingVerdict(criteria)

      if (previousPending !== nextPending) {
        emit.verdictPendingChanged({ forkId: event.forkId, pending: nextPending })
      }

      const shellResolved = criteria.filter((c) =>
        c.type === 'shell' && (c.lifecycle._tag === 'passed' || c.lifecycle._tag === 'failed')
      ).length

      const allShellPassed = fork.shellCount === 0
        ? true
        : shellResolved === fork.shellCount && criteria.filter((c) => c.type === 'shell').every((c) => c.lifecycle._tag === 'passed')

      return {
        ...fork,
        criteria,
        shellResolved,
        allShellPassed,
        resolved: false,
      }
    },
  },

  globalEventHandlers: {
    phase_criteria_verdict: ({ event, state, emit }) => {
      const targetForkId = event.parentForkId
      const fork = state.forks.get(targetForkId)
      if (!fork || fork.resolved) return state

      const previousPending = hasPendingVerdict(fork.criteria)
      const next = applyVerdict(fork, event)
      const nextPending = hasPendingVerdict(next.criteria)
      if (previousPending !== nextPending) {
        emit.verdictPendingChanged({ forkId: targetForkId, pending: nextPending })
      }

      const failed = next.criteria.some((c) => c.lifecycle._tag === 'failed')
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

      const resolvedCount = next.criteria.filter((c) => c.lifecycle._tag === 'passed' || c.lifecycle._tag === 'failed').length
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
