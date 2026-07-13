import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { queryImageTool } from '../tools/query-image'
import { QueryImageStateSchema, type QueryImageState } from './tool-state'

export { QueryImageStateSchema, type QueryImageState } from './tool-state'

const initial: Omit<QueryImageState, 'phase' | 'errorMessage'> = {
  path: Option.none(),
  query: Option.none(),
}

export const queryImageModel = defineStateModel(queryImageTool)({
  state: QueryImageStateSchema,
  initial,
  reduce: (state, event): QueryImageState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'path') {
          return { ...state, phase: 'streaming', path: Option.some(Option.getOrElse(state.path, () => '') + event.delta) }
        }
        if (event.field === 'query') {
          return { ...state, phase: 'streaming', query: Option.some(Option.getOrElse(state.query, () => '') + event.delta) }
        }
        return state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          path: typeof event.input.path === 'string' ? Option.some(event.input.path) : state.path,
          query: typeof event.input.query === 'string' ? Option.some(event.input.query) : state.query,
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
