import { TextAttributes } from '@opentui/core'
import { createToolDisplay } from '../types'
import { useTheme } from '../../hooks/use-theme'

type PhaseSubmitState = {
  phase?: string
}

function getErrorText(result: any): string | undefined {
  if (!result) return undefined
  if (result.status === 'error') return result.message ?? 'phase submit failed'
  if (result.status === 'success' && typeof result.output === 'string') {
    const text = result.output.trim()
    if (text.startsWith('phase submit failed:') || text.startsWith('phase criteria failed:')) {
      return text
    }
  }
  return undefined
}

function getSuccessText(result: any): string {
  if (result?.status === 'success' && typeof result.output === 'string') {
    const output = result.output.trim().toLowerCase()
    if (output.includes('workflow completed')) return 'Phase submitted (workflow completed)'
    if (output.includes('workflow advanced')) return 'Phase submitted'
  }
  return 'Phase submitted'
}

export const phaseSubmitDisplay = createToolDisplay<PhaseSubmitState>(['phase-submit', 'workflow-submit'], {
  render: ({ state, result }) => {
    const theme = useTheme()
    const isRunning = state.phase === 'streaming' || state.phase === 'executing'
    const errorText = getErrorText(result)

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
        <span style={{ fg: theme.foreground }}>{getSuccessText(result)}</span>
      </text>
    )
  },
  summary: (state) => {
    if (state.phase === 'streaming' || state.phase === 'executing') return 'Submitting phase'
    return 'Submitted phase'
  },
})
