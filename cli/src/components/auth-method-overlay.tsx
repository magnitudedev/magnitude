import { memo, useState, useCallback } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { WizardHeader, type WizardMode } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'
import type { AuthMethodDef } from '@magnitudedev/agent'

interface AuthMethodOverlayProps {
  providerName: string
  methods: AuthMethodDef[]
  selectedIndex: number
  onSelectedIndexChange: (index: number) => void
  onSelect: (methodIndex: number) => void
  onHoverIndex?: (index: number) => void
  onBack: () => void
  wizardMode?: WizardMode
}

export const AuthMethodOverlay = memo(function AuthMethodOverlay({
  providerName,
  methods,
  selectedIndex,
  onSelectedIndexChange,
  onSelect,
  onHoverIndex,
  onBack,
  wizardMode,
}: AuthMethodOverlayProps) {
  const theme = useTheme()
  const [headerBackHovered, setHeaderBackHovered] = useState(false)
  const [backHovered, setBackHovered] = useState(false)

  useKeyboard(useCallback((key: KeyEvent) => {
    const plain = !key.ctrl && !key.meta && !key.option

    if (key.name === 'escape') {
      key.preventDefault()
      wizardMode?.onBack?.() ?? onBack()
      return
    }

    if (key.ctrl && key.name === 's' && !key.meta && !key.option && !key.shift && wizardMode?.onSkip) {
      key.preventDefault()
      wizardMode.onSkip()
      return
    }

    if (methods.length === 0) return

    if (key.name === 'up' && plain) {
      key.preventDefault()
      onSelectedIndexChange(Math.max(0, selectedIndex - 1))
      return
    }

    if (key.name === 'down' && plain) {
      key.preventDefault()
      onSelectedIndexChange(Math.min(methods.length - 1, selectedIndex + 1))
      return
    }

    if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
      key.preventDefault()
      onSelect(selectedIndex)
      return
    }

    if (!key.defaultPrevented) {
      key.preventDefault()
    }
  }, [onBack, wizardMode, methods.length, selectedIndex, onSelectedIndexChange, onSelect]))

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
              <Button
                onClick={onBack}
                onMouseOver={() => setHeaderBackHovered(true)}
                onMouseOut={() => setHeaderBackHovered(false)}
              >
                <text style={{ fg: headerBackHovered ? theme.foreground : theme.muted }} attributes={TextAttributes.UNDERLINE}>Back</text>
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
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back (Esc)</text>
            </box>
          </Button>
        </box>
      )}
    </box>
  )
})
