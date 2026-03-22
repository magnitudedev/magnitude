import { existsSync } from 'node:fs'
import { resolveTemplates } from './template'
import type { Criteria, FieldError, Phase, ValidationResult, WorkflowAction, WorkflowSkill } from './types'

export interface WorkflowState {
  readonly skill: WorkflowSkill
  readonly currentPhaseIndex: number
  readonly submissions: ReadonlyMap<string, string>
  readonly phaseStatuses: readonly PhaseStatus[]
  readonly status: 'active' | 'completed'
}

export type PhaseStatus = 'pending' | 'active' | 'awaiting-criteria' | 'completed'

function withResolvedCriteria(
  criteria: readonly Criteria[] | undefined,
  submissions: ReadonlyMap<string, string>,
): readonly Criteria[] {
  if (!criteria) return []
  return criteria.map((item) => {
    if (item.type === 'shell-succeed') {
      return { ...item, command: resolveTemplates(item.command, submissions) }
    }
    if (item.type === 'user-approval') {
      return { ...item, message: resolveTemplates(item.message, submissions) }
    }
    return { ...item, prompt: resolveTemplates(item.prompt, submissions) }
  })
}

function resolvePhaseTemplates(phase: Phase, submissions: ReadonlyMap<string, string>): Phase {
  return {
    ...phase,
    criteria: withResolvedCriteria(phase.criteria, submissions),
    hooks: phase.hooks
      ? {
          onStart: phase.hooks.onStart ? resolveTemplates(phase.hooks.onStart, submissions) : undefined,
          onSubmit: phase.hooks.onSubmit ? resolveTemplates(phase.hooks.onSubmit, submissions) : undefined,
          onAccept: phase.hooks.onAccept ? resolveTemplates(phase.hooks.onAccept, submissions) : undefined,
          onReject: phase.hooks.onReject ? resolveTemplates(phase.hooks.onReject, submissions) : undefined,
        }
      : undefined,
  }
}

function advanceState(state: WorkflowState): WorkflowState {
  const nextIndex = state.currentPhaseIndex + 1
  const phaseCount = state.skill.phases.length
  const currentStatuses = [...state.phaseStatuses]

  if (phaseCount === 0 || nextIndex >= phaseCount) {
    if (phaseCount > 0 && state.currentPhaseIndex < phaseCount) {
      currentStatuses[state.currentPhaseIndex] = 'completed'
    }

    return {
      ...state,
      currentPhaseIndex: phaseCount,
      phaseStatuses: currentStatuses,
      status: 'completed',
    }
  }

  currentStatuses[state.currentPhaseIndex] = 'completed'
  currentStatuses[nextIndex] = 'active'

  return {
    ...state,
    currentPhaseIndex: nextIndex,
    phaseStatuses: currentStatuses,
  }
}

export function createWorkflowState(skill: WorkflowSkill): WorkflowState {
  const phaseStatuses =
    skill.phases.length === 0
      ? []
      : skill.phases.map((_, idx) => (idx === 0 ? 'active' : 'pending' as const))

  return {
    skill,
    currentPhaseIndex: 0,
    submissions: new Map(),
    phaseStatuses,
    status: skill.phases.length === 0 ? 'completed' : 'active',
  }
}

export function getCurrentPhase(state: WorkflowState): Phase | null {
  if (state.status === 'completed') return null
  const phase = state.skill.phases[state.currentPhaseIndex]
  if (!phase) return null
  return resolvePhaseTemplates(phase, state.submissions)
}

export function getCurrentPrompt(state: WorkflowState): string {
  const phase = getCurrentPhase(state)
  if (!phase) return ''

  const resolvedPrompt = resolveTemplates(phase.prompt, state.submissions)
  if (state.currentPhaseIndex === 0) {
    const preamble = state.skill.preamble.trim()
    return preamble ? `${preamble}\n\n${resolvedPrompt}`.trim() : resolvedPrompt
  }

  return resolvedPrompt
}

export function validateFields(state: WorkflowState, fields: ReadonlyMap<string, string>): ValidationResult {
  const phase = getCurrentPhase(state)
  if (!phase) return { valid: true }

  const errors: FieldError[] = []
  for (const field of phase.submit?.fields ?? []) {
    if (field.type !== 'file') continue
    const value = fields.get(field.name)?.trim()
    if (!value) {
      errors.push({ type: 'missing', name: field.name })
    } else if (!existsSync(value)) {
      errors.push({ type: 'file-not-found', name: field.name, path: value })
    }
  }

  if (errors.length > 0) {
    return { valid: false, errors }
  }

  return { valid: true }
}

export function reduce(state: WorkflowState, action: WorkflowAction): WorkflowState {
  switch (action.type) {
    case 'submit': {
      const phase = getCurrentPhase(state)
      if (!phase) return state

      const updatedSubmissions = new Map(state.submissions)
      for (const [fieldName, value] of action.fields.entries()) {
        updatedSubmissions.set(`${phase.name}.${fieldName}`, value)
      }

      const nextState: WorkflowState = {
        ...state,
        submissions: updatedSubmissions,
      }

      if ((phase.criteria ?? []).length === 0) {
        return nextState
      }

      const phaseStatuses = [...state.phaseStatuses]
      phaseStatuses[state.currentPhaseIndex] = 'awaiting-criteria'
      return {
        ...nextState,
        phaseStatuses,
      }
    }

    case 'advance':
      return advanceState(state)

    case 'criteria-failed': {
      const phaseStatuses = [...state.phaseStatuses]
      phaseStatuses[state.currentPhaseIndex] = 'active'
      return {
        ...state,
        phaseStatuses,
      }
    }
  }
}
