import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { skillTool } from '../tools/skill-tool'

export interface SkillActivationState extends BaseState {
  toolKey: 'skill'
  skillName?: string
  skillPath?: string
  contentPreview?: string
  errorDetail?: string
}

const initial: Omit<SkillActivationState, 'phase' | 'toolKey'> = {
  skillName: undefined,
  skillPath: undefined,
  contentPreview: undefined,
  errorDetail: undefined,
}

export const skillActivationModel = defineStateModel('skill', skillTool)({
  initial,
  reduce: (state, event): SkillActivationState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorDetail: undefined }
      case 'ToolInputFieldChunk':
        return event.field === 'name'
          ? { ...state, phase: 'streaming', skillName: (state.skillName ?? '') + event.delta }
          : state
      case 'ToolInputReady':
        return { ...state, phase: 'streaming', skillName: event.input.name as string | undefined }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success': {
            const output = event.result.output as { content: string; skillPath?: string }
            const content = output.content
            return {
              ...state,
              phase: 'completed',
              skillPath: output.skillPath ?? state.skillPath,
              contentPreview: content.length > 200 ? content.slice(0, 200) + '…' : content,
            }
          }
          case 'Error':
            return { ...state, phase: 'error', errorDetail: event.result.error }
          case 'Rejected':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
        }
      }
      case 'ToolInputParseError':
        return { ...state, phase: 'error', errorDetail: event.error.detail }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
