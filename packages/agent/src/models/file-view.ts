import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { viewTool } from '../tools/fs'

export interface FileViewState extends BaseState {
  toolKey: 'fileView'
  path?: string
}

const initial: Omit<FileViewState, 'phase' | 'toolKey'> = {
  path: undefined,
}

export const fileViewModel = defineStateModel('fileView', viewTool)({
  initial,
  reduce: (state, event): FileViewState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', path: event.streaming.path?.value ?? state.path }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
        return { ...state, phase: 'executing' }
      case 'parseError':
        return { ...state, phase: 'error' }
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
