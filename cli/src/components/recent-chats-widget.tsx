import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { RecentChatEntry } from './recent-chat-entry'
import type { RecentChat } from '../data/recent-chats'

interface RecentChatsWidgetProps {
  chats: RecentChat[]
  loading?: boolean
  selectedIndex: number
  onSelect: (chat: RecentChat) => void
  onHoverIndex?: (index: number) => void
  onOpenAll?: () => void
  isNavigationActive: boolean
}

export const RecentChatsWidget = memo(function RecentChatsWidget({
  chats,
  loading,
  selectedIndex,
  onSelect,
  onHoverIndex,
  onOpenAll,
  isNavigationActive,
}: RecentChatsWidgetProps) {
  const theme = useTheme()

  if (chats.length === 0 && loading) {
    return (
      <box style={{ flexDirection: 'column', paddingTop: 1, paddingBottom: 1 }}>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>Loading recent chats...</span>
        </text>
      </box>
    )
  }

  if (chats.length === 0) return null

  return (
    <box style={{ flexDirection: 'column', paddingTop: 1, paddingBottom: 1 }}>
      <box style={{ flexDirection: 'row', paddingBottom: 1 }}>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>Recent conversations · </span>
        </text>
        <Button onClick={onOpenAll}>
          <text style={{ fg: theme.muted }} attributes={TextAttributes.UNDERLINE}>See all</text>
        </Button>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>{' '}(Ctrl+R)</span>
        </text>
      </box>
      {chats.map((chat, index) => (
        <RecentChatEntry
          key={chat.id}
          chat={chat}
          isSelected={isNavigationActive && index === selectedIndex}
          onSelect={onSelect}
          onHover={() => onHoverIndex?.(index)}
        />
      ))}
    </box>
  )
})
