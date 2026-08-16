import React from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { LOGO_LINES } from '@magnitudedev/client-common'

export function AnimatedLogo() {
  const theme = useTheme()

  return (
    <box style={{ flexDirection: 'column' }}>
      {LOGO_LINES.map((line, i) => (
        <text key={i} style={{ fg: theme.accent }}>{line}</text>
      ))}
      <text>{' '}</text>
      <text style={{ fg: theme.text.body }} attributes={TextAttributes.BOLD}>Magnitude</text>
      <text style={{ fg: theme.text.supporting }}>Your actually local agent</text>
    </box>
  )
}
