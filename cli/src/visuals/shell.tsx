/**
 * Shell Tool Visual — Renderer
 *
 * Pure render function for shell tool visual state.
 * State is pre-reduced by DisplayProjection via the shell reducer.
 */

import { TextAttributes } from '@opentui/core'
import type { ShellState } from '@magnitudedev/agent'
import { render } from './define'
import { Button } from '../components/button'
import { ShimmerText } from '../components/shimmer-text'
import { useTheme } from '../hooks/use-theme'
import { shortenCommandPreview } from '../utils/strings'

// =============================================================================
// Constants
// =============================================================================

const SHIMMER_INTERVAL_MS = 160
const RESULT_TRUNCATE_LEN = 80
const MAX_COMMAND_DISPLAY_LEN = 80

// =============================================================================
// Helpers
// =============================================================================

function truncateLine(text: string, max: number): string {
  if (!text) return ''
  const firstLine = text.split('\n').find(l => l.trim() !== '') ?? ''
  if (firstLine.length > max) return firstLine.slice(0, max - 3) + '...'
  return firstLine
}

// =============================================================================
// Render
// =============================================================================

export function shellLiveText({ state }: { state: ShellState }): string {
  const command = state.command.trim()
  if (state.phase !== 'done') return command.length > 0 ? `$ ${command}` : 'Running shell command'
  if (state.done?.kind === 'error') return command.length > 0 ? `Shell error: $ ${command}` : 'Shell error'
  if (state.done?.kind === 'rejected') return command.length > 0 ? `Rejected: $ ${command}` : 'Shell command rejected'
  return command.length > 0 ? `$ ${command}` : 'Ran shell command'
}

export const shellRender = render<ShellState>(({ state, isExpanded, onToggle }) => {
  const theme = useTheme()
  const { phase, command, done } = state

  const isRunning = phase !== 'done'
  const isRejected = done?.kind === 'rejected'
  const isError = done?.kind === 'error'
  const isSuccess = done?.kind === 'success'
  const isInterrupted = done?.kind === 'interrupted'

  // Determine failed state and result text for preview
  const isFailed = (isSuccess && done.exitCode !== 0) || isError
  const resultPreview = isSuccess
    ? (isFailed ? (done.stderr || done.stdout) : (done.stdout || done.stderr))
    : isError
      ? done.message
      : ''

  // Full expanded text
  const fullResultText = isSuccess
    ? [done.stdout, isFailed ? done.stderr : ''].filter(Boolean).join('\n').replace(/^\n+/, '').trimEnd()
    : isError
      ? done.message
      : ''

  return (
    <box style={{ flexDirection: 'column' }}>
      {/* Command line */}
      {isInterrupted ? (
        <text>
          <span style={{ fg: theme.muted }}>{'$ '}</span>
          <span style={{ fg: theme.foreground }}>
            {shortenCommandPreview(command, MAX_COMMAND_DISPLAY_LEN)}
          </span>
          <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{' · Interrupted'}</span>
        </text>
      ) : (
        <Button onClick={onToggle}>
          <text>
            <span style={{ fg: theme.muted }}>{'$ '}</span>
            <span style={{ fg: theme.foreground }}>
              {isExpanded ? command : shortenCommandPreview(command, MAX_COMMAND_DISPLAY_LEN)}
            </span>
            {isRunning ? (
              <>
                {'  '}
                <ShimmerText text="running..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : isRejected ? (
              done.systemReason
                ? <><span style={{ fg: theme.error }}>{' · System Rejected'}</span><span style={{ fg: theme.muted }}>{` (${done.systemReason})`}</span></>
                : <span style={{ fg: theme.error }}>{' · User Rejected'}</span>
            ) : (isSuccess || isError) ? (
              <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                {isExpanded ? ' · (collapse)' : ' · (expand)'}
              </span>
            ) : null}
          </text>
        </Button>
      )}

      {/* Error preview — collapsed second line when failed */}
      {isFailed && !isExpanded && resultPreview.trim() && (
        <text style={{ fg: theme.error }} attributes={TextAttributes.DIM}>
          {'✗ '}{truncateLine(resultPreview, RESULT_TRUNCATE_LEN)}
        </text>
      )}

      {/* Expanded result */}
      {isExpanded && (isSuccess || isError) && fullResultText.trim() && (
        <text style={{ fg: isFailed ? theme.error : theme.muted }} attributes={TextAttributes.DIM}>
          {isFailed ? '✗ ' : ''}{fullResultText}
        </text>
      )}
    </box>
  )
})
