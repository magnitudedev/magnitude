import { TextAttributes } from '@opentui/core'

import { useTheme } from '../hooks/use-theme'
import { BOX_CHARS } from '../utils/ui-constants'

export type DiffHunkProps = {
  contextBefore?: string[]
  removedLines: string[]
  addedLines: string[]
  contextAfter?: string[]
  streamingCursor?: boolean
  maxHeight?: number
}

export function DiffHunk({
  contextBefore = [],
  removedLines,
  addedLines,
  contextAfter = [],
  streamingCursor = false,
  maxHeight = 12,
}: DiffHunkProps) {
  const theme = useTheme()

  return (
    <box
      style={{
        borderStyle: 'single',
        borderColor: theme.border || theme.muted,
        customBorderChars: BOX_CHARS,
        height: maxHeight,
      }}
    >
      <scrollbox
        stickyScroll
        stickyStart="bottom"
        scrollX={false}
        scrollbarOptions={{ visible: false }}
        verticalScrollbarOptions={{ visible: false }}
        style={{
          flexGrow: 1,
          rootOptions: { flexGrow: 1, backgroundColor: 'transparent' },
          wrapperOptions: { border: false, backgroundColor: 'transparent', paddingLeft: 1, paddingRight: 1 },
          contentOptions: { justifyContent: 'flex-start' },
        }}
      >
        <box style={{ flexDirection: 'column' }}>
          {contextBefore.map((line, index) => (
            <text key={`context-before-${index}`} style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
              {line}
            </text>
          ))}

          {removedLines.map((line, index) => (
            <text key={`removed-${index}`} style={{ fg: theme.error }}>
              {`- ${line}`}
            </text>
          ))}

          {addedLines.map((line, index) => (
            <text key={`added-${index}`} style={{ fg: theme.syntax.string }}>
              {`+ ${line}${streamingCursor && index === addedLines.length - 1 ? '▍' : ''}`}
            </text>
          ))}

          {contextAfter.map((line, index) => (
            <text key={`context-after-${index}`} style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
              {line}
            </text>
          ))}
        </box>
      </scrollbox>
    </box>
  )
}
