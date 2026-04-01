import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { webFetchTool, webFetchXmlBinding } from '../tools/web-fetch-tool'

export interface WebFetchState extends BaseState {
  toolKey: 'webFetch'
  url?: string
  errorDetail?: string
}

const initial: Omit<WebFetchState, 'phase' | 'toolKey'> = {
  url: undefined,
  errorDetail: undefined,
}

export const webFetchModel = defineStateModel('webFetch', {
  tool: webFetchTool,
  binding: webFetchXmlBinding,
})({
  initial,
  reduce: (state, event): WebFetchState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming', errorDetail: undefined }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', url: event.streaming.url?.value ?? state.url }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
        return { ...state, phase: 'executing' }
      case 'parseError':
        return { ...state, phase: 'error', errorDetail: event.error }
      case 'completed':
        return { ...state, phase: 'completed', url: event.output.url }
      case 'error':
        return { ...state, phase: 'error', errorDetail: event.error.message }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
