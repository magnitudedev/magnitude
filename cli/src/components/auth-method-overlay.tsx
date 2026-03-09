import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { WizardHeader, type WizardMode } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'
import type { AuthMethodDef } from '@magnitudedev/agent'

interface AuthMethodOverlayProps {
  providerName: string
  methods: AuthMethodDef[]
  selectedIndex: number
  onSelect: (methodIndex: number) => void
  onHoverIndex?: (index: number) => void
  onBack: () => void
  wizardMode?: WizardMode
}

export const AuthMethodOverlay = memo(function AuthMethodOverlay({
  providerName,
  methods,
  selectedIndex,
  onSelect,
  onHoverIndex,
  onBack,
  wizardMode,
}: AuthMethodOverlayProps) {
  const theme = useTheme()
  const [backHovered, setBackHovered] = useState(false)

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      {wizardMode ? (
        <WizardHeader
          stepLabel={wizardMode.stepLabel}
          subtitle={wizardMode.subtitle}
          onSkip={wizardMode.onSkip}
          theme={theme}
        />
      ) : (
        <>
          {/* Header */}
          <box style={{
            flexDirection: 'row',
            paddingLeft: 2,
            paddingRight: 2,
            paddingTop: 1,
            paddingBottom: 1,
            flexShrink: 0,
          }}>
            <text style={{ fg: theme.primary, flexGrow: 1 }}>
              <span attributes={TextAttributes.BOLD}>Connect {providerName}</span>
            </text>
            <box style={{ flexDirection: 'row' }}>
              <Button onClick={onBack}>
                <text style={{ fg: theme.muted }} attributes={TextAttributes.UNDERLINE}>Back</text>
              </Button>
              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>{' '}(Esc)  |  Enter to select</span>
              </text>
            </box>
          </box>

          {/* Divider */}
          <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
            <text style={{ fg: theme.border }}>
              {'─'.repeat(80)}
            </text>
          </box>
        </>
      )}

      {/* Auth method list */}
      <box style={{ paddingLeft: 1, paddingRight: 1, paddingTop: 1, flexGrow: 1 }}>
        <text style={{ fg: theme.muted, paddingLeft: 1, paddingBottom: 1 }}>
          Select auth method:
        </text>
        {methods.map((method, index) => {
          const isSelected = index === selectedIndex

          return (
            <Button
              key={index}
              onClick={() => onSelect(index)}
              onMouseOver={() => onHoverIndex?.(index)}
              style={{
                flexDirection: 'row',
                paddingLeft: 1,
                paddingRight: 1,
                backgroundColor: isSelected ? theme.surface : undefined,
              }}
            >
              <text style={{ fg: isSelected ? theme.primary : theme.foreground }}>
                {isSelected ? '> ' : '  '}
                {method.label}
                {method.type === 'oauth-pkce' && (
                  <span>{' '.repeat(3)}<span style={{ fg: theme.error }}>Warning:</span> Use this method at your own risk. There have been reports of users getting banned.</span>
                )}
              </text>
            </Button>
          )
        })}
      </box>

      {wizardMode && (
        <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
          <Button onClick={wizardMode.onBack} onMouseOver={() => setBackHovered(true)} onMouseOut={() => setBackHovered(false)}>
            <box style={{
              borderStyle: 'single',
              borderColor: backHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back</text>
            </box>
          </Button>
        </box>
      )}
    </box>
  )
})
