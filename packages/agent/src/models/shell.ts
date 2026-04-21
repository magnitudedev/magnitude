import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { shellTool } from '../tools/shell'

export interface ShellState extends BaseState {
  toolKey: 'shell'
  command: string
  done: 'completed' | null
  exitCode?: number
  stdout?: string
  stderr?: string
  errorMessage?: string
}

const initial: Omit<ShellState, 'phase' | 'toolKey'> = {
  command: '',
  done: null,
  exitCode: undefined,
  stdout: undefined,
  stderr: undefined,
  errorMessage: undefined,
}

export const shellModel = defineStateModel('shell', shellTool)({
  initial,
  reduce: (state, event): ShellState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', done: null }
      case 'ToolInputFieldChunk':
        return event.field === 'command'
          ? { ...state, command: state.command + event.delta }
          : state
      case 'ToolInputReady':
        return { ...state, command: event.input.command, phase: 'streaming' }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return {
              ...state,
              phase: 'completed',
              done: 'completed',
              exitCode: event.result.output.exitCode,
              stdout: event.result.output.stdout,
              stderr: event.result.output.stderr,
              errorMessage: undefined,
            }
          case 'Error':
            return { ...state, phase: 'error', errorMessage: event.result.error }
          case 'Rejected':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
        }
      }
      case 'ToolParseError':
        return { ...state, phase: 'error', errorMessage: event.error }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
