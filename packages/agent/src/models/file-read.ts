import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { readTool, readXmlBinding } from '../tools/fs'

export interface FileReadState extends BaseState {
  path?: string
  lineCount?: number
  errorDetail?: string
}

const initial: Omit<FileReadState, 'phase'> = {
  path: undefined,
  lineCount: undefined,
  errorDetail: undefined,
}

export const fileReadModel = defineStateModel({
  tool: readTool,
  binding: readXmlBinding,
})({
  initial,
  reduce: (state, event): FileReadState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming', errorDetail: undefined }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', path: event.streaming.fields.path ?? state.path }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'completed':
        return { ...state, phase: 'completed', lineCount: event.output.split('\n').length }
      case 'error':
        return { ...state, phase: 'error', errorDetail: event.error.message }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
