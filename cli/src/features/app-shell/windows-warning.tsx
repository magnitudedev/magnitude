import { memo, useCallback } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../../hooks/use-theme'

interface WindowsWarningScreenProps {
  onExit: () => void
}

export const WindowsWarningScreen = memo(function WindowsWarningScreen({
  onExit,
}: WindowsWarningScreenProps) {
  const theme = useTheme()

  useKeyboard(
    useCallback(
      (key: KeyEvent) => {
        if (key.name === 'escape') {
          key.preventDefault()
          onExit()
          return
        }
        if (key.ctrl && key.name === 'c' && !key.meta && !key.option) {
          key.preventDefault()
          onExit()
          return
        }
      },
      [onExit],
    ),
  )

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <box
        style={{
          flexDirection: 'row',
          paddingLeft: 2,
          paddingRight: 2,
          paddingTop: 1,
          paddingBottom: 1,
          flexShrink: 0,
        }}
      >
        <text style={{ fg: theme.error, flexGrow: 1 }}>
          <span attributes={TextAttributes.BOLD}>
            Native Windows is not supported
          </span>
        </text>
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>{'─'.repeat(80)}</text>
      </box>

      {/* Body */}
      <box
        style={{
          paddingLeft: 2,
          paddingRight: 2,
          paddingTop: 1,
          flexGrow: 1,
          flexDirection: 'column',
        }}
      >
        <box style={{ paddingBottom: 1 }}>
          <text style={{ fg: theme.foreground }}>
            Magnitude requires a Unix-like shell environment and does not run
            on native Windows. Please use WSL (Windows Subsystem for Linux) instead.
          </text>
        </box>

        <box style={{ paddingBottom: 1, flexDirection: 'column' }}>
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.muted }}>1. Open PowerShell as Administrator and run:</text>
          </box>
          <box style={{ paddingLeft: 2, paddingBottom: 1 }}>
            <text style={{ fg: theme.foreground }}>
              <span attributes={TextAttributes.BOLD}>wsl --install</span>
            </text>
          </box>
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.muted }}>2. Restart your computer if prompted, then open a terminal and run:</text>
          </box>
          <box style={{ paddingLeft: 2, paddingBottom: 1 }}>
            <text style={{ fg: theme.foreground }}>
              <span attributes={TextAttributes.BOLD}>wsl</span>
            </text>
          </box>
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.muted }}>3. Inside the WSL shell, install Magnitude:</text>
          </box>
          <box style={{ paddingLeft: 2, paddingBottom: 1 }}>
            <text style={{ fg: theme.foreground }}>
              <span attributes={TextAttributes.BOLD}>npm i -g @magnitudedev/cli</span>
            </text>
          </box>
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.muted }}>4. Then run it from your project directory:</text>
          </box>
          <box style={{ paddingLeft: 2, paddingBottom: 1 }}>
            <text style={{ fg: theme.foreground }}>
              <span attributes={TextAttributes.BOLD}>magnitude</span>
            </text>
          </box>
        </box>
      </box>

      {/* Footer */}
      <box
        style={{
          paddingLeft: 2,
          paddingRight: 2,
          paddingTop: 1,
          paddingBottom: 1,
          flexShrink: 0,
        }}
      >
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>
            Press Esc or Ctrl+C to exit
          </span>
        </text>
      </box>
    </box>
  )
})

export type { WindowsWarningScreenProps }
