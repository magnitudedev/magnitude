import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { webSearchTool, webSearchXmlBinding } from '../tools/web-search-tool'

type Source = { title: string; url: string }

export interface WebSearchState extends BaseState {
  toolKey: 'webSearch'
  query?: string
  sources: Source[]
  errorDetail?: string
}

const initial: Omit<WebSearchState, 'phase' | 'toolKey'> = {
  query: undefined,
  sources: [],
  errorDetail: undefined,
}

export const webSearchModel = defineStateModel('webSearch', {
  tool: webSearchTool,
  binding: webSearchXmlBinding,
})({
  initial,
  reduce: (state, event): WebSearchState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming', errorDetail: undefined }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', query: event.streaming.query?.value ?? state.query }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
        return { ...state, phase: 'executing' }
      case 'parseError':
        return { ...state, phase: 'error', errorDetail: event.error }
      case 'completed':
        return { ...state, phase: 'completed', sources: [...(event.output.sources as Source[])] }
      case 'error':
        return { ...state, phase: 'error', errorDetail: event.error.message }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
