/**
 * Skill Activation Model
 *
 * State model for the skill tool — tracks skill activation requests.
 */

import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { skillTool, skillXmlBinding } from '../tools/skill-tool'

export interface SkillActivationState extends BaseState {
  toolKey: 'skill'
  skillName?: string
  skillPath?: string      // resolved skill file path
  contentPreview?: string
  errorDetail?: string
}

const initial: Omit<SkillActivationState, 'phase' | 'toolKey'> = {
  skillName: undefined,
  skillPath: undefined,
  contentPreview: undefined,
  errorDetail: undefined,
}

export const skillActivationModel = defineStateModel('skill', {
  tool: skillTool,
  binding: skillXmlBinding,
})({
  initial,
  reduce: (state, event): SkillActivationState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming', errorDetail: undefined }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', skillName: event.streaming.name?.value ?? state.skillName }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
        return { ...state, phase: 'executing' }
      case 'parseError':
        return { ...state, phase: 'error', errorDetail: event.error }
      case 'completed':
        // Preview first 200 chars of content
        const content = event.output.content as string
        return {
          ...state,
          phase: 'completed',
          skillPath: event.output.skillPath ?? state.skillPath,
          contentPreview: content.length > 200 ? content.slice(0, 200) + '…' : content,
        }
      case 'error':
        return { ...state, phase: 'error', errorDetail: event.error.message }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
