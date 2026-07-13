import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { shellTool } from '../tools/shell'
import { ShellStateSchema, type ShellState } from './tool-state'

export { ShellStateSchema, type ShellState } from './tool-state'

const initial: Omit<ShellState, 'phase' | 'errorMessage'> = {
  command: '',
  done: Option.none(),
  exitCode: Option.none(),
  stdout: Option.none(),
  stderr: Option.none(),
  stdoutPath: Option.none(),
  stderrPath: Option.none(),
  pid: Option.none(),
  partialStdout: '',
  partialStderr: '',
}

export const shellModel = defineStateModel(shellTool)({
  state: ShellStateSchema,
  initial,
  reduce: (state, event): ShellState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', done: Option.none() }
      case 'ToolInputFieldChunk':
        return event.field === 'command'
          ? { ...state, command: state.command + event.delta }
          : state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing', command: typeof event.input.command === 'string' ? event.input.command : state.command }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success': {
            const output = event.result.output
            if (output.mode === 'completed') {
              return {
                ...state,
                phase: 'completed',
                done: Option.some('completed' as const),
                exitCode: Option.some(output.exitCode),
                stdout: Option.some(output.stdout),
                stderr: Option.some(output.stderr),
                partialStdout: output.stdout,
                partialStderr: output.stderr,
                errorMessage: Option.none(),
              }
            } else {
              // detached
              return {
                ...state,
                phase: 'completed',
                done: Option.some('detached' as const),
                pid: Option.some(output.pid),
                stdoutPath: Option.some(output.stdoutPath),
                stderrPath: Option.some(output.stderrPath),
                errorMessage: Option.none(),
              }
            }
          }
          case 'Error':
            return { ...state, phase: 'error', errorMessage: Option.some(event.result.error.message) }
          case 'Denied':
            return { ...state, phase: 'rejected', errorMessage: Option.some(String(event.result.denial)) }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          default:
            return state
        }
      }
      case 'ToolEmission': {
        const v = event.value as { type: string; stdout?: string; stderr?: string }
        if (v.type === 'shell_output') {
          return {
            ...state,
            partialStdout: state.partialStdout + (v.stdout ?? ''),
            partialStderr: state.partialStderr + (v.stderr ?? ''),
          }
        }
        return state
      }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorMessage: Option.some(event.issue.message) }
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
