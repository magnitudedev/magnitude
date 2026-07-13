import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { viewTool } from '../tools/fs'
import { FileViewStateSchema, type FileViewState } from './tool-state'

export { FileViewStateSchema, type FileViewState } from './tool-state'

const initial: Omit<FileViewState, 'phase' | 'errorMessage'> = {
  path: Option.none(),
}

export const fileViewModel = defineStateModel(viewTool)({
  state: FileViewStateSchema,
  initial,
  reduce: (state, event): FileViewState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return event.field === 'path'
          ? { ...state, phase: 'streaming', path: Option.some(Option.getOrElse(state.path, () => '') + event.delta) }
          : state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          path: typeof event.input.path === 'string' ? Option.some(event.input.path) : state.path,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed' }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Denied':
            return { ...state, phase: 'rejected', errorMessage: Option.some(String(event.result.denial)) }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          default:
            return state
        }
      }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorMessage: Option.some(event.issue.message) }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
