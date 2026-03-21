import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { writeTool, writeXmlBinding } from '../tools/fs'

export interface FileWriteState extends BaseState {
  path?: string
  body: string
  charCount: number
  lineCount: number
}

const initial: Omit<FileWriteState, 'phase'> = {
  path: undefined,
  body: '',
  charCount: 0,
  lineCount: 0,
}

export const fileWriteModel = defineStateModel({
  tool: writeTool,
  binding: writeXmlBinding,
})({
  initial,
  reduce: (state, event): FileWriteState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady': {
        const content = event.streaming.body ?? ''
        return {
          ...state,
          phase: 'streaming',
          path: event.streaming.fields.path ?? state.path,
          body: event.streaming.body ?? '',
          charCount: content.length,
          lineCount: content.length > 0 ? content.split('\n').length : 0,
        }
      }
      case 'executionStarted':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'emission':
        return event.value.type === 'write_stats'
          ? { ...state, phase: 'executing', path: event.value.path, lineCount: event.value.linesWritten }
          : state
      case 'completed':
        return { ...state, phase: 'completed' }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
