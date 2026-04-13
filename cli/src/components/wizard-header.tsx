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
  skipDisabled?: boolean
  theme: Record<string, any>
}

export const WizardHeader = memo(function WizardHeader({
  stepLabel,
  subtitle,
  onSkip,
  skipDisabled = false,
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
        <Button
          onClick={skipDisabled ? undefined : onSkip}
          onMouseOver={() => !skipDisabled && setSkipHovered(true)}
          onMouseOut={() => setSkipHovered(false)}
        >
          <box style={{
            borderStyle: 'single',
            borderColor: skipDisabled ? theme.border : (skipHovered ? theme.primary : theme.border),
            customBorderChars: BOX_CHARS,
            paddingLeft: 1,
            paddingRight: 1,
            marginLeft: 1,
            ...(skipDisabled ? { opacity: 0.6 } : {}),
          }}>
            <text style={{ fg: skipDisabled ? theme.border : (skipHovered ? theme.primary : theme.muted) }}>Skip (Ctrl+S)</text>
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
