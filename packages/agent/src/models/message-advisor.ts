import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { messageAdvisorTool } from '../tools/advisor'
import { MessageAdvisorStateSchema, type MessageAdvisorState } from './tool-state'

export { MessageAdvisorStateSchema, type MessageAdvisorState } from './tool-state'

const PREVIEW_CHARS = 240

function append(current: Option.Option<string>, delta: string): Option.Option<string> {
  return Option.some(`${Option.getOrElse(current, () => '')}${delta}`)
}

function preview(text: string): string {
  const trimmed = text.trim()
  return trimmed.length > PREVIEW_CHARS
    ? `${trimmed.slice(0, PREVIEW_CHARS)}...`
    : trimmed
}

export const messageAdvisorModel = defineStateModel(messageAdvisorTool)({
  state: MessageAdvisorStateSchema,
  initial: {
    message: Option.none(),
    responsePreview: Option.none(),
  },
  reduce: (state, event): MessageAdvisorState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorMessage: Option.none() }
      case 'ToolInputFieldChunk':
        return event.field === 'message'
          ? { ...state, phase: 'streaming', message: append(state.message, event.delta) }
          : state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          message: typeof event.input.message === 'string' ? Option.some(event.input.message) : state.message,
          errorMessage: Option.none(),
        }
      case 'ToolExecutionEnded':
        switch (event.result._tag) {
          case 'Success':
            return {
              ...state,
              phase: 'completed',
              responsePreview: Option.some(preview(event.result.output)),
            }
          case 'Error':
            return { ...state, phase: 'error', errorMessage: Option.some(event.result.error.message) }
          case 'Denied':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          case 'InputRejected':
            return { ...state, phase: 'error', errorMessage: Option.some(event.result.issue.message) }
        }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorMessage: Option.some(event.issue.message) }
      case 'ToolInputReady':
      case 'ToolInputFieldComplete':
      case 'ToolEmission':
      default:
        return state
    }
  },
})
