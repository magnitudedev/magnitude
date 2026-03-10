import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'

import { BOX_CHARS } from '../utils/ui-constants'
import { Button } from './button'
import { shortenCommandPreview } from '../utils/strings'
import type { BashResult } from '../utils/bash-executor'

const BASH_BOX_CHARS = { ...BOX_CHARS, vertical: '▎' }
const MAX_PREVIEW_LINES = 5
const MAX_COMMAND_DISPLAY_LEN = 80


interface BashOutputProps {
  result: BashResult
}

export const BashOutput = memo(function BashOutput({ result }: BashOutputProps) {
  const theme = useTheme()
  const [isExpanded, setIsExpanded] = useState(false)

  // Combine all output lines for truncation logic
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

  const totalLines = allLines.length
  const needsOutputTruncation = totalLines > MAX_PREVIEW_LINES
  const commandNeedsTruncation = shortenCommandPreview(result.command, MAX_COMMAND_DISPLAY_LEN) !== result.command
  const needsExpand = needsOutputTruncation || commandNeedsTruncation
  const visibleLines = (!needsOutputTruncation || isExpanded)
    ? allLines
    : allLines.slice(0, MAX_PREVIEW_LINES)
  const hiddenCount = totalLines - MAX_PREVIEW_LINES

  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <box
        style={{
          flexDirection: 'column',
          borderStyle: 'single',
          border: ['left'],
          borderColor: theme.primary,
          customBorderChars: BASH_BOX_CHARS,
          paddingLeft: 1,
          paddingRight: 1,
          paddingTop: 0,
          paddingBottom: 0,
        }}
      >
        {/* Command line */}
        <text style={{ fg: theme.foreground, wrapMode: isExpanded ? 'word' : 'none' }}>
          <span fg={theme.primary} attributes={TextAttributes.BOLD}>$ </span>
          <span attributes={TextAttributes.BOLD}>{isExpanded ? result.command : shortenCommandPreview(result.command, MAX_COMMAND_DISPLAY_LEN)}</span>
        </text>

        {/* Output lines */}
        {visibleLines.length > 0 && (
          <text style={{ fg: theme.foreground, wrapMode: 'word' }}>
            {visibleLines.map((entry, i) => (
              <span key={i} fg={entry.color}>
                {entry.line}{i < visibleLines.length - 1 ? '\n' : ''}
              </span>
            ))}
          </text>
        )}

        {/* Expand/collapse button */}
        {needsExpand && (
          <Button onClick={() => setIsExpanded(prev => !prev)}>
            <text style={{ fg: theme.muted }}>
              {isExpanded
                ? '▾ Collapse'
                : needsOutputTruncation
                  ? `▸ Show ${hiddenCount} more lines`
                  : '▸ Show all'}
            </text>
          </Button>
        )}
      </box>
    </box>
  )
})
