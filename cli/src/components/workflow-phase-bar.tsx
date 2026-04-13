import React, { useState, useEffect } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

interface WorkflowPhaseBarProps {
  state: {
    skillName: string
    phases: Array<{ name: string; status: 'pending' | 'active' | 'verifying' | 'completed' }>
  }
}

export function WorkflowPhaseBar({ state }: WorkflowPhaseBarProps) {
  const theme = useTheme()
  const isVerifying = state.phases.some((p) => p.status === 'verifying')

  const [spinnerIndex, setSpinnerIndex] = useState(0)
  useEffect(() => {
    if (!isVerifying) return
    const interval = setInterval(() => setSpinnerIndex((i) => (i + 1) % SPINNER_FRAMES.length), 80)
    return () => clearInterval(interval)
  }, [isVerifying])

  return (
    <box style={{ flexDirection: 'column', flexShrink: 0, width: '100%', borderStyle: 'single', border: ['top', 'bottom'], borderColor: theme.border }}>
      <box style={{ flexDirection: 'row', paddingLeft: 1, paddingRight: 1 }}>
        <text style={{ fg: theme.muted }}>{`${state.skillName}: `}</text>
        {state.phases.map((phase, i) => (
          <box key={`${phase.name}-${i}`} style={{ flexDirection: 'row' }}>
            {i > 0 ? <text style={{ fg: theme.muted }}>{'  '}</text> : null}
            {phase.status === 'completed' ? (
              <>
                <text style={{ fg: theme.success }}>{'✓'}</text>
                <text style={{ fg: theme.muted }}>{` ${phase.name}`}</text>
              </>
            ) : phase.status === 'verifying' ? (
              <>
                <text style={{ fg: theme.primary }}>{SPINNER_FRAMES[spinnerIndex]}</text>
                <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{` ${phase.name}`}</text>
              </>
            ) : phase.status === 'active' ? (
              <>
                <text style={{ fg: theme.primary }}>{'→'}</text>
                <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{` ${phase.name}`}</text>
              </>
            ) : (
              <>
                <text style={{ fg: theme.muted }}>{'·'}</text>
                <text style={{ fg: theme.muted }}>{` ${phase.name}`}</text>
              </>
            )}
          </box>
        ))}
      </box>
    </box>
  )
}
