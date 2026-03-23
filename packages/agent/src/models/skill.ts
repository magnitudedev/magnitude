import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { skillTool, skillXmlBinding } from '../tools/skill'

export interface SkillState extends BaseState {
  toolKey: 'skill'
  name?: string
}

const initial: Omit<SkillState, 'phase' | 'toolKey'> = {
  name: undefined,
}

export const skillModel = defineStateModel('skill', {
  tool: skillTool,
  binding: skillXmlBinding,
})({
  initial,
  reduce: (state, event): SkillState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', name: event.streaming.fields.name ?? state.name }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'completed':
        return { ...state, phase: 'completed' }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
