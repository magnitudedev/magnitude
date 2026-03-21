/**
 * Shell Tool Visual — Renderer
 *
 * Pure render function for shell tool visual state.
 * State is pre-reduced by DisplayProjection via the shell reducer.
 */

import type { ShellState } from '@magnitudedev/agent/src/models'

// =============================================================================
// Render
// =============================================================================

export function shellLiveText({ state }: { state: ShellState }): string {
  const command = state.command.trim()
  if (state.phase === 'streaming' || state.phase === 'executing') return command.length > 0 ? `$ ${command}` : 'Running shell command'
  if (state.phase === 'error') return command.length > 0 ? `Shell error: $ ${command}` : 'Shell error'
  if (state.phase === 'rejected') return command.length > 0 ? `Rejected: $ ${command}` : 'Shell command rejected'
  if (state.done === 'detached') return 'Detached shell'
  return command.length > 0 ? `$ ${command}` : 'Ran shell command'
}


