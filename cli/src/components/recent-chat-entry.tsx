import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { formatRelativeTime, type RecentChat } from '../data/recent-chats'

interface RecentChatEntryProps {
  chat: RecentChat
  isSelected: boolean
  onSelect: (chat: RecentChat) => void
  onHover?: () => void
}

export const RecentChatEntry = memo(function RecentChatEntry({
  chat,
  isSelected,
  onSelect,
  onHover,
}: RecentChatEntryProps) {
  const theme = useTheme()

  const timeStr = formatRelativeTime(chat.timestamp)
  const messageLabel = chat.messageCount === 1 ? '1 message' : `${chat.messageCount} messages`
  const rightSide = `${messageLabel} · ${timeStr}`
  const maxTitleLen = 120 - rightSide.length - 6
  const displayTitle = chat.title.length > maxTitleLen
    ? chat.title.slice(0, maxTitleLen) + '…'
    : chat.title

  return (
    <Button
      onClick={() => onSelect(chat)}
      onMouseOver={onHover}
      style={{
        flexDirection: 'row',
        paddingLeft: 1,
        paddingRight: 1,
        backgroundColor: isSelected ? theme.surface : undefined,
      }}
    >
      <text style={{ fg: isSelected ? theme.primary : theme.foreground, flexGrow: 1 }}>
        <span attributes={TextAttributes.BOLD}>
          {isSelected ? '> ' : '  '}{displayTitle}
        </span>
      </text>
      <text style={{ fg: theme.muted, flexShrink: 0 }}>
        {' '}{messageLabel} · {timeStr}
      </text>
    </Button>
  )
})
