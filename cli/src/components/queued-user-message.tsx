import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { BOX_CHARS } from '../utils/ui-constants'

const QUEUED_BOX_CHARS = { ...BOX_CHARS, vertical: '┃' }

interface QueuedUserMessageProps {
  content: string
}

export const QueuedUserMessage = memo(function QueuedUserMessage({ content }: QueuedUserMessageProps) {
  const theme = useTheme()

  return (
    <box style={{ flexDirection: 'column', position: 'relative', marginBottom: 1 }}>
      {/* Same structure as UserMessage but dimmed */}
      <box
        style={{
          borderStyle: 'single',
          border: ['left'],
          borderColor: theme.muted,
          customBorderChars: QUEUED_BOX_CHARS,
        }}
      >
        <box
          style={{
            flexDirection: 'column',
            backgroundColor: theme.userMessageBg,
            paddingTop: 1,
            paddingBottom: 1,
            paddingLeft: 1,
            paddingRight: 2,
            flexGrow: 1,
          }}
        >
          <text style={{ fg: theme.foreground, wrapMode: 'word', flexGrow: 1 }} attributes={TextAttributes.DIM}>
{content}
          </text>
        </box>
      </box>
    </box>
  )
})
