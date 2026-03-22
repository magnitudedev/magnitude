import React, { useState, useEffect } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

interface WorkflowPhaseBarProps {
  state: {
    skillName: string
    phases: Array<{ name: string; status: 'pending' | 'active' | 'verifying' | 'completed' }>
    criteria: Array<{ index: number; name: string; type: string; status: string; reason?: string }> | null
  }
}

export function WorkflowPhaseBar({ state }: WorkflowPhaseBarProps) {
  const theme = useTheme()
  const isVerifying = state.phases.some((p) => p.status === 'verifying')
  const showCriteria = Boolean(state.criteria?.length) && isVerifying

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
      {showCriteria ? (
        <box style={{ flexDirection: 'row', paddingLeft: 3, paddingRight: 1 }}>
          {(state.criteria ?? []).map((c, i) => (
            <box key={c.index} style={{ flexDirection: 'row' }}>
              {i > 0 ? <text style={{ fg: theme.muted }}>{' · '}</text> : null}
              <text style={{ fg: theme.foreground }}>{c.name}</text>
              {c.status === 'running' ? (
                <text style={{ fg: theme.info }}>{' ● running'}</text>
              ) : c.status === 'passed' ? (
                <text style={{ fg: theme.success }}>{' ✓'}</text>
              ) : c.status === 'failed' ? (
                <text style={{ fg: theme.error }}>{' ✗'}</text>
              ) : (
                <text style={{ fg: theme.muted }}>{' ◌ pending'}</text>
              )}
            </box>
          ))}
        </box>
      ) : null}
    </box>
  )
}
