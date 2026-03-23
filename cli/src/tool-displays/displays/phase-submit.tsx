import { TextAttributes } from '@opentui/core'
import { createToolDisplay } from '../types'
import { useTheme } from '../../hooks/use-theme'

type PhaseSubmitState = {
  phase?: string
  output?: string
  errorMessage?: string
}

function getErrorText(state: PhaseSubmitState): string | undefined {
  if (typeof state.errorMessage === 'string' && state.errorMessage.length > 0) {
    return state.errorMessage
  }
  if (typeof state.output === 'string') {
    const text = state.output.trim()
    if (text.startsWith('phase submit failed:') || text.startsWith('phase criteria failed:')) {
      return text
    }
  }
  return undefined
}

function getSuccessText(state: PhaseSubmitState): string {
  if (typeof state.output === 'string') {
    const output = state.output.trim().toLowerCase()
    if (output.includes('workflow completed')) return 'Phase submitted (workflow completed)'
    if (output.includes('workflow advanced')) return 'Phase submitted'
  }
  return 'Phase submitted'
}

export const phaseSubmitDisplay = createToolDisplay<PhaseSubmitState>({
  render: ({ state }) => {
    const theme = useTheme()
    const isRunning = state.phase === 'streaming' || state.phase === 'executing'
    const errorText = getErrorText(state)

    if (isRunning) {
      return (
        <text>
          <span style={{ fg: theme.info }}>{'* '}</span>
          <span style={{ fg: theme.muted }}>Submitting phase...</span>
        </text>
      )
    }

    if (errorText) {
      return (
        <text style={{ fg: theme.error }} attributes={TextAttributes.DIM}>
          {'✗ '}{errorText}
        </text>
      )
    }

    return (
      <text>
        <span style={{ fg: theme.primary }}>{'* '}</span>
        <span style={{ fg: theme.foreground }}>{getSuccessText(state)}</span>
      </text>
    )
  },
  summary: (state) => {
    if (state.phase === 'streaming' || state.phase === 'executing') return 'Submitting phase'
    return 'Submitted phase'
  },
})
