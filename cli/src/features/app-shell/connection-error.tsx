import React from 'react'
import { useKeyboard } from '@opentui/react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../hooks/use-theme'

interface FatalErrorScreenProps {
  error: string
  invariantViolation?: boolean
  onRetry: () => void
  onQuit: () => void
}

export const FatalErrorScreen = ({ error, invariantViolation = false, onRetry, onQuit }: FatalErrorScreenProps) => {
  const theme = useTheme()

  useKeyboard((key) => {
    if (key.defaultPrevented) return

    if (key.name === 'r' || key.name === 'R') {
      key.preventDefault()
      onRetry()
      return
    }

    if (key.name === 'q' || key.name === 'Q') {
      key.preventDefault()
      onQuit()
    }
  })

  const title = invariantViolation
    ? 'An unexpected error occurred'
    : 'Failed to connect to Magnitude daemon'

  const body = invariantViolation
    ? `${error}\n\nPlease report this issue.`
    : error

  return (
    <box
      style={{
        flexDirection: 'column',
        flexGrow: 1,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <text style={{ fg: theme.error }} attributes={TextAttributes.BOLD}>
        {title}
      </text>

      <box style={{ marginTop: 1, maxWidth: 80, paddingLeft: 2, paddingRight: 2 }}>
        <text style={{ fg: theme.foreground }}>{body}</text>
      </box>

      <box
        style={{
          marginTop: 2,
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <text style={{ fg: theme.muted }}>Press R to retry</text>
        <text style={{ fg: theme.muted }}>Press Q to quit</text>
      </box>
    </box>
  )
}
