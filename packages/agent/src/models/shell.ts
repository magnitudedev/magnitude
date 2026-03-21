import { defineStateModel, type BaseState, type Phase } from '@magnitudedev/tools'
import { shellTool, shellXmlBinding } from '../tools/shell'

export interface ShellState extends BaseState {
  command: string
  done: string | null
}

export const shellModel = defineStateModel({
  tool: shellTool,
  binding: shellXmlBinding,
})({
  initial: {
    command: '',
    done: null as string | null,
  },
  reduce: (state, event) => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' as Phase, done: null }
      case 'inputUpdated':
        return { ...state, command: event.streaming.body ?? state.command }
      case 'inputReady':
        return { ...state, command: event.input.command }
      case 'executionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'completed':
        return { ...state, phase: 'completed' as Phase, done: event.output.mode }
      case 'error':
        return { ...state, phase: 'error' as Phase }
      case 'rejected':
      case 'approvalRejected':
        return { ...state, phase: 'rejected' as Phase }
      case 'interrupted':
        return { ...state, phase: 'interrupted' as Phase }
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'parseError':
        return state
    }
  },
})
