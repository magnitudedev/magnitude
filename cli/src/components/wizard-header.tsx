import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { Button } from './button'
import { BOX_CHARS } from '../utils/ui-constants'

export interface WizardMode {
  stepLabel: string
  subtitle: string
  onSkip: () => void
  onBack: () => void
}

interface WizardHeaderProps {
  stepLabel: string
  subtitle: string
  onSkip: () => void
  theme: Record<string, any>
}

export const WizardHeader = memo(function WizardHeader({
  stepLabel,
  subtitle,
  onSkip,
  theme,
}: WizardHeaderProps) {
  const [skipHovered, setSkipHovered] = useState(false)

  return (
    <>
      {/* Header row */}
      <box style={{
        flexDirection: 'row',
        alignItems: 'center',
        paddingLeft: 2,
        paddingRight: 2,
        paddingTop: 1,
        flexShrink: 0,
      }}>
        <text style={{ fg: theme.primary, flexGrow: 1 }}>
          <span attributes={TextAttributes.BOLD}>Welcome to Magnitude!</span>
        </text>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>{stepLabel}</span>
        </text>
        <Button onClick={onSkip} onMouseOver={() => setSkipHovered(true)} onMouseOut={() => setSkipHovered(false)}>
          <box style={{
            borderStyle: 'single',
            borderColor: skipHovered ? theme.primary : theme.border,
            customBorderChars: BOX_CHARS,
            paddingLeft: 1,
            paddingRight: 1,
            marginLeft: 1,
          }}>
            <text style={{ fg: skipHovered ? theme.primary : theme.muted }}>Skip (Esc)</text>
          </box>
        </Button>
      </box>

      {/* Subtitle */}
      <box style={{ paddingLeft: 2, paddingRight: 2, flexShrink: 0, maxWidth: 100 }}>
        <text style={{ fg: theme.foreground }}>
          {subtitle}
        </text>
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>{'─'.repeat(100)}</text>
      </box>
    </>
  )
})
