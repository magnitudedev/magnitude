import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { webSearchTool } from '../tools/web-search'

export interface WebSearchState extends BaseState {
  toolKey: 'webSearch'
  query?: string
  sources?: readonly { title: string; url: string }[]
  errorDetail?: string
}

const initial: Omit<WebSearchState, 'phase' | 'toolKey'> = {
  query: undefined,
  sources: undefined,
  errorDetail: undefined,
}

export const webSearchModel = defineStateModel('webSearch', webSearchTool)({
  initial,
  reduce: (state, event): WebSearchState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorDetail: undefined }
      case 'ToolInputFieldChunk':
        return event.field === 'query'
          ? { ...state, phase: 'streaming', query: (state.query ?? '') + event.delta }
          : state
      case 'ToolInputReady':
        return { ...state, phase: 'streaming', query: event.input.query }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', sources: (event.result.output as { sources?: readonly { title: string; url: string }[] }).sources ?? [] }
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
