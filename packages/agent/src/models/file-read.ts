import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { readTool } from '../tools/fs'
import { FileReadStateSchema, type FileReadState } from './tool-state'

export { FileReadStateSchema, type FileReadState } from './tool-state'

const initial: Omit<FileReadState, 'phase' | 'errorMessage'> = {
  path: Option.none(),
  lineCount: Option.none(),
  offset: Option.none(),
  limit: Option.none(),
  errorDetail: Option.none(),
}

export const fileReadModel = defineStateModel(readTool)({
  state: FileReadStateSchema,
  initial,
  reduce: (state, event): FileReadState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorDetail: Option.none() }
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
          offset: typeof event.input.offset === 'number' ? Option.some(event.input.offset) : state.offset,
          limit: typeof event.input.limit === 'number' ? Option.some(event.input.limit) : state.limit,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', lineCount: Option.some(event.result.output.split('\n').length) }
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
