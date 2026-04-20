import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { webFetchTool } from '../tools/web-fetch-tool'

export interface WebFetchState extends BaseState {
  toolKey: 'webFetch'
  url?: string
  errorDetail?: string
}

const initial: Omit<WebFetchState, 'phase' | 'toolKey'> = {
  url: undefined,
  errorDetail: undefined,
}

export const webFetchModel = defineStateModel('webFetch', webFetchTool)({
  initial,
  reduce: (state, event): WebFetchState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorDetail: undefined }
      case 'ToolInputFieldChunk':
        return event.field === 'url'
          ? { ...state, phase: 'streaming', url: (state.url ?? '') + event.delta }
          : state
      case 'ToolInputReady':
        return { ...state, phase: 'streaming', url: event.input.url }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', url: (event.result.output as { url: string }).url }
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
