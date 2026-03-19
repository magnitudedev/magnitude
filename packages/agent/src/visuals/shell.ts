/**
 * Shell Tool Visual Reducer
 *
 * State machine for shell command execution.
 * Processes streaming ToolCallEvents from the shell tool.
 */

import type { XmlToolResult } from '@magnitudedev/xml-act'
import { shellTool } from '../tools/shell'
import { defineToolReducer } from './define'

// =============================================================================
// State
// =============================================================================

type ShellPhase = 'streaming' | 'executing' | 'done'

type DoneVariant =
  | { readonly kind: 'success'; readonly stdout: string; readonly stderr: string; readonly exitCode: number }
  | { readonly kind: 'detached'; readonly pid: number; readonly stdout: string; readonly stderr: string }
  | { readonly kind: 'error'; readonly message: string }
  | { readonly kind: 'rejected'; readonly systemReason: string | null }
  | { readonly kind: 'interrupted' }

export interface ShellState {
  readonly phase: ShellPhase
  /** Command text accumulated from body chunks during streaming */
  readonly command: string
  /** Final result, populated when phase is 'done' */
  readonly done: DoneVariant | null
}

// =============================================================================
// Helpers
// =============================================================================

function resolveResult(
  result: XmlToolResult<
    | { mode: 'completed'; stdout: string; stderr: string; exitCode: number }
    | { mode: 'detached'; pid: number; stdout: string; stderr: string }
  >
): DoneVariant {
  switch (result._tag) {
    case 'Success': {
      const output = result.output
      if (output.mode === 'detached') {
        return { kind: 'detached', pid: output.pid, stdout: output.stdout, stderr: output.stderr }
      }
      return {
        kind: 'success',
        stdout: output.stdout,
        stderr: output.stderr,
        exitCode: output.exitCode,
      }
    }
    case 'Error':
      return { kind: 'error', message: result.error }
    case 'Rejected': {
      const rejection = result.rejection
      if (rejection && typeof rejection === 'object' && '_tag' in rejection) {
        const tagged = rejection as { _tag: string; reason?: string }
        if (tagged._tag !== 'UserRejection') {
          return { kind: 'rejected', systemReason: tagged.reason ?? tagged._tag }
        }
      }
      return { kind: 'rejected', systemReason: null }
    }
    case 'Interrupted':
      return { kind: 'interrupted' }
  }
}

// =============================================================================
// Reducer
// =============================================================================

export const shellReducer = defineToolReducer({
  tool: shellTool,
  toolKey: 'shell',
  cluster: 'shell',

  initial: {
    phase: 'streaming',
    command: '',
    done: null,
  } satisfies ShellState,

  reduce(state, event): ShellState {
    switch (event._tag) {
      case 'ToolInputBodyChunk':
        return { ...state, command: state.command + event.text }

      case 'ToolInputReady':
        return { ...state, command: event.input.command }

      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing', command: event.input.command }

      case 'ToolExecutionEnded':
        return { ...state, phase: 'done', done: resolveResult(event.result) }

      default:
        return state
    }
  },
})
