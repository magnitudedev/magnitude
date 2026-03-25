import { defineStateModel, type BaseState, type Phase } from '@magnitudedev/tools'
import { shellTool, shellXmlBinding } from '../tools/shell'

export interface ShellState extends BaseState {
  toolKey: 'shell'
  command: string
  done: string | null
  exitCode?: number
  stdout?: string
  stderr?: string
  pid?: number
  completionMode?: string
  errorMessage?: string
  rejectionReason?: string
}

export const shellModel = defineStateModel('shell', {
  tool: shellTool,
  binding: shellXmlBinding,
})({
  initial: {
    command: '',
    done: null as string | null,
    exitCode: undefined as number | undefined,
    stdout: undefined as string | undefined,
    stderr: undefined as string | undefined,
    pid: undefined as number | undefined,
    completionMode: undefined as string | undefined,
    errorMessage: undefined as string | undefined,
    rejectionReason: undefined as string | undefined,
  },
  reduce: (state, event) => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' as Phase, done: null }
      case 'inputUpdated':
        return { ...state, command: event.streaming.command?.value ?? state.command }
      case 'inputReady':
        return { ...state, command: event.input.command }
      case 'executionStarted':
        return { ...state, phase: 'executing' as Phase }
      case 'completed':
        return {
          ...state,
          phase: 'completed' as Phase,
          done: event.output.mode,
          completionMode: event.output.mode,
          exitCode: event.output.mode === 'completed' ? event.output.exitCode : undefined,
          stdout: event.output.stdout,
          stderr: event.output.stderr,
          pid: event.output.mode === 'detached' ? event.output.pid : undefined,
          errorMessage: undefined,
          rejectionReason: undefined,
        }
      case 'error':
        return { ...state, phase: 'error' as Phase, errorMessage: event.error.message }
      case 'rejected':
        return {
          ...state,
          phase: 'rejected' as Phase,
          rejectionReason: 'rejected',
        }
      case 'approvalRejected':
        return { ...state, phase: 'rejected' as Phase, rejectionReason: 'approvalRejected' }
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
