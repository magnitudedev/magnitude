import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../../hooks/use-theme'
import { orange } from '../../../utils/theme'

import { BOX_CHARS } from '../../../utils/ui-constants'
import type { UserBashCommandMessage } from '@magnitudedev/sdk'

const BASH_ACCENT = orange[400]
const BASH_BOX_CHARS = { ...BOX_CHARS, vertical: '▎' }

interface BashOutputProps {
  result: UserBashCommandMessage
}

export const BashOutput = memo(function BashOutput({ result }: BashOutputProps) {
  const theme = useTheme()

  // Combine all output lines
  const outputParts: Array<{ text: string; color: string }> = []
  if (result.stdout.length > 0) {
    outputParts.push({ text: result.stdout, color: theme.foreground })
  }
  if (result.stderr.length > 0) {
    outputParts.push({ text: result.stderr, color: theme.error })
  }

  const allLines: Array<{ line: string; color: string }> = []
  for (const part of outputParts) {
    for (const line of part.text.split('\n')) {
      allLines.push({ line, color: part.color })
    }
  }

  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <box
        style={{
          flexDirection: 'column',
          borderStyle: 'single',
          border: ['left'],
          borderColor: BASH_ACCENT,
          customBorderChars: BASH_BOX_CHARS,
          paddingLeft: 1,
          paddingRight: 1,
          paddingTop: 0,
          paddingBottom: 0,
        }}
      >
        {/* Command line */}
        <text style={{ fg: theme.foreground, wrapMode: 'word' }}>
          <span fg={BASH_ACCENT} attributes={TextAttributes.BOLD}>$ </span>
          <span attributes={TextAttributes.BOLD}>{result.command}</span>
          <span fg={result.exitCode === 0 ? theme.success : theme.error}>
            {result.exitCode === 0 ? ' ✓' : ` ✗ Exit ${result.exitCode}`}
          </span>
        </text>

        {/* Output lines */}
        {allLines.length > 0 && (
          <text style={{ fg: theme.foreground, wrapMode: 'word' }}>
            {allLines.map((entry, i) => (
              <span key={i} fg={entry.color}>
                {entry.line}{i < allLines.length - 1 ? '\n' : ''}
              </span>
            ))}
          </text>
        )}
      </box>
    </box>
  )
})
