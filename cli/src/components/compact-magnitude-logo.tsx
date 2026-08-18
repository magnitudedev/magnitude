import { useTheme } from '../hooks/use-theme'

/** Compact Magnitude mark at its exact 9×5 terminal-cell geometry. */
export const COMPACT_MAGNITUDE_LOGO_WIDTH = 9
export const COMPACT_MAGNITUDE_LOGO_HEIGHT = 5

export const COMPACT_MAGNITUDE_LOGO_LINES = [
  '┏━━━━━━━┓',
  '┃ ↗   ↗ ┃',
  '┃   ◡   ┃',
  '┗━━━┳━━━┛',
  '   ━┻━',
] as const

export function CompactMagnitudeLogo() {
  const theme = useTheme()
  return (
    <box style={{
      width: COMPACT_MAGNITUDE_LOGO_WIDTH,
      height: COMPACT_MAGNITUDE_LOGO_HEIGHT,
      flexShrink: 0,
      flexDirection: 'column',
    }}>
      {COMPACT_MAGNITUDE_LOGO_LINES.map((line, index) => (
        <text key={index} style={{ fg: theme.accent }}>{line}</text>
      ))}
    </box>
  )
}
