import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { shellBgTool, shellBgXmlBinding } from '../tools/shell-bg'

export interface ShellBgState extends BaseState {
  toolKey: 'shellBg'
}

const initial: Omit<ShellBgState, 'phase' | 'toolKey'> = {}

export const shellBgModel = defineStateModel('shellBg', {
  tool: shellBgTool,
  binding: shellBgXmlBinding,
})({
  initial,
  reduce: (state, event): ShellBgState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming' }
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
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
