import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../../hooks/use-theme'
import { UserMessageFrame } from './user-message-frame'

interface QueuedUserMessageProps {
  content: string
}

export const QueuedUserMessage = memo(function QueuedUserMessage({ content }: QueuedUserMessageProps) {
  const theme = useTheme()

  return (
    <box style={{ flexDirection: 'column', position: 'relative', marginBottom: 1 }}>
      <UserMessageFrame
        borderColor={theme.muted}
        backgroundColor={theme.userMessageBg}
      >
        <text style={{ fg: theme.foreground, wrapMode: 'word', flexGrow: 1 }} attributes={TextAttributes.DIM}>
{content}
        </text>
      </UserMessageFrame>
    </box>
  )
})
