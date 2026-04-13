import { Projection } from '@magnitudedev/event-core'
import { createWorkflowState, type WorkflowState } from '@magnitudedev/skills'
import type { AppEvent } from '../events'

export interface WorkflowCriteriaState {
  readonly workflowState: WorkflowState | null
  readonly skillName: string
  readonly phases: Array<{
    name: string
    status: 'pending' | 'active'
  }>
}

const initialFork: WorkflowCriteriaState = {
  workflowState: null,
  skillName: '',
  phases: [],
}

export const WorkflowProjection = Projection.defineForked<AppEvent, WorkflowCriteriaState>()({
  name: 'Workflow',

  initialFork,

  signals: {},

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

    skill_completed: ({ fork }) => ({
      ...fork,
      workflowState: null,
      skillName: '',
      phases: [],
    }),
  },

  globalEventHandlers: {},
})
