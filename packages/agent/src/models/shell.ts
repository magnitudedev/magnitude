import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { shellTool, shellXmlBinding } from '../tools/shell'

export interface ShellState extends BaseState {
  toolKey: 'shell'
  command: string
  done: 'completed' | null
  exitCode?: number
  stdout?: string
  stderr?: string
  errorMessage?: string
  rejectionReason?: string
}

export const shellModel = defineStateModel('shell', {
  tool: shellTool,
  binding: shellXmlBinding,
})({
  initial: {
    command: '',
    done: null,
    exitCode: undefined,
    stdout: undefined,
    stderr: undefined,
    errorMessage: undefined,
    rejectionReason: undefined,
  },
  reduce: (state, event) => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming', done: null }
      case 'inputUpdated':
        return { ...state, command: event.streaming.command?.value ?? state.command }
      case 'inputReady':
        return { ...state, command: event.input.command }
      case 'executionStarted':
        return { ...state, phase: 'executing' }
      case 'completed':
        return {
          ...state,
          phase: 'completed',
          done: 'completed',
          exitCode: event.output.exitCode,
          stdout: event.output.stdout,
          stderr: event.output.stderr,
          errorMessage: undefined,
          rejectionReason: undefined,
        }
      case 'error':
        return { ...state, phase: 'error', errorMessage: event.error.message }
      case 'rejected':
        return {
          ...state,
          phase: 'rejected',
          rejectionReason: 'rejected',
        }
      case 'approvalRejected':
        return { ...state, phase: 'rejected', rejectionReason: 'approvalRejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'parseError':
        return state
    }
  },
})
