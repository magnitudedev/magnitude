import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { webSearchTool } from '../tools/web-search'
import { WebSearchSourceSchema, WebSearchStateSchema, type WebSearchSource, type WebSearchState } from './tool-state'

export { WebSearchSourceSchema, WebSearchStateSchema, type WebSearchSource, type WebSearchState } from './tool-state'

const initial: Omit<WebSearchState, 'phase' | 'errorMessage'> = {
  query: Option.none(),
  sources: Option.none(),
  errorDetail: Option.none(),
}

export const webSearchModel = defineStateModel(webSearchTool)({
  state: WebSearchStateSchema,
  initial,
  reduce: (state, event): WebSearchState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorDetail: Option.none() }
      case 'ToolInputFieldChunk':
        return event.field === 'query'
          ? { ...state, phase: 'streaming', query: Option.some(Option.getOrElse(state.query, () => '') + event.delta) }
          : state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          query: typeof event.input.query === 'string' ? Option.some(event.input.query) : state.query,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return {
              ...state,
              phase: 'completed',
              sources: Array.isArray(event.result.output.sources) ? Option.some(event.result.output.sources) : Option.some([]),
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
