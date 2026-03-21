import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { clickTool, clickXmlBinding } from '../tools/browser-tools'

export interface BrowserActionState extends BaseState {
  label?: string
  detail?: string
  errorDetail?: string
}

const initial: Omit<BrowserActionState, 'phase'> = {
  label: undefined,
  detail: undefined,
  errorDetail: undefined,
}

const formatDetail = (fields: Record<string, unknown>): string | undefined => {
  const entries = Object.entries(fields).filter(([, v]) => v !== undefined && v !== null && v !== '')
  if (entries.length === 0) return undefined
  return entries.map(([k, v]) => `${k}=${String(v)}`).join(', ')
}

export const browserActionModel = defineStateModel({
  tool: clickTool,
  binding: clickXmlBinding,
})({
  initial,
  reduce: (state, event): BrowserActionState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming', errorDetail: undefined }
      case 'inputUpdated':
      case 'inputReady': {
        const fields = event.streaming.fields as Record<string, unknown>
        return {
          ...state,
          phase: 'streaming',
          label: 'Browser action',
          detail: formatDetail(fields) ?? event.streaming.body ?? undefined,
        }
      }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'completed':
        return { ...state, phase: 'completed' }
      case 'error':
        return { ...state, phase: 'error', errorDetail: event.error.message }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
