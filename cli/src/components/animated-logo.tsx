import React from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { LOGO_LINES } from '../utils/ascii-logo'

export function AnimatedLogo() {
  const theme = useTheme()

  return (
    <box style={{ flexDirection: 'column' }}>
      {LOGO_LINES.map((line, i) => (
        <text key={i} style={{ fg: theme.primary }}>{line}</text>
      ))}
      <text>{' '}</text>
      <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Magnitude</text>
      <text style={{ fg: theme.muted }}>Open Source Coding Agent</text>
    </box>
  )
}
