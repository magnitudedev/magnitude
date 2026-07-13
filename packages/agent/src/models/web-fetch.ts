import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { webFetchTool } from '../tools/web-fetch-tool'
import { WebFetchStateSchema, type WebFetchState } from './tool-state'

export { WebFetchStateSchema, type WebFetchState } from './tool-state'

const initial: Omit<WebFetchState, 'phase' | 'errorMessage'> = {
  url: Option.none(),
  errorDetail: Option.none(),
}

export const webFetchModel = defineStateModel(webFetchTool)({
  state: WebFetchStateSchema,
  initial,
  reduce: (state, event): WebFetchState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorDetail: Option.none() }
      case 'ToolInputFieldChunk':
        return event.field === 'url'
          ? { ...state, phase: 'streaming', url: Option.some(Option.getOrElse(state.url, () => '') + event.delta) }
          : state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          url: typeof event.input.url === 'string' ? Option.some(event.input.url) : state.url,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', url: Option.some(event.result.output.url) }
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
