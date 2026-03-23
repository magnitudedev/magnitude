import { createToolDisplay } from '../types'
import { useTheme } from '../../hooks/use-theme'

type PhaseVerdictState = {
  phase?: string
}

export const phaseVerdictDisplay = createToolDisplay<PhaseVerdictState>({
  render: ({ state }) => {
    const theme = useTheme()
    const isRunning = state.phase === 'streaming' || state.phase === 'executing'

    return (
      <text>
        <span style={{ fg: isRunning ? theme.info : theme.primary }}>{'* '}</span>
        <span style={{ fg: isRunning ? theme.muted : theme.foreground }}>
          {isRunning ? 'Submitting verdict...' : 'Verdict submitted'}
        </span>
      </text>
    )
  },
  summary: (state) => {
    return state.phase === 'streaming' || state.phase === 'executing' ? 'Submitting verdict' : 'Verdict submitted'
  },
})
