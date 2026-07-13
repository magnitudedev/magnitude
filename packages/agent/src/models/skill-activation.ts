import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { skillTool } from '../tools/skill-tool'
import { SkillActivationStateSchema, type SkillActivationState } from './tool-state'

export { SkillActivationStateSchema, type SkillActivationState } from './tool-state'

const initial: Omit<SkillActivationState, 'phase' | 'errorMessage'> = {
  skillName: Option.none(),
  skillPath: Option.none(),
  contentPreview: Option.none(),
  errorDetail: Option.none(),
}

export const skillActivationModel = defineStateModel(skillTool)({
  state: SkillActivationStateSchema,
  initial,
  reduce: (state, event): SkillActivationState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorDetail: Option.none() }
      case 'ToolInputFieldChunk':
        return event.field === 'name'
          ? { ...state, phase: 'streaming', skillName: Option.some(Option.getOrElse(state.skillName, () => '') + event.delta) }
          : state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          skillName: typeof event.input.name === 'string' ? Option.some(event.input.name) : state.skillName,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success': {
            const output = event.result.output
            const content = output.content
            return {
              ...state,
              phase: 'completed',
              skillPath: typeof output.skillPath === 'string' ? Option.some(output.skillPath) : state.skillPath,
              contentPreview: Option.some(content.length > 200 ? content.slice(0, 200) + '…' : content),
            }
          }
          case 'Error':
            return { ...state, phase: 'error', errorDetail: Option.some(event.result.error.message) }
          case 'Denied':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          default:
            return state
        }
      }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorDetail: Option.some(event.issue.message) }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
