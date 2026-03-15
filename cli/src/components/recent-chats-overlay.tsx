import { memo, useState, useCallback } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { RecentChatEntry } from './recent-chat-entry'
import type { RecentChat } from '../data/recent-chats'

interface RecentChatsOverlayProps {
  chats: RecentChat[]
  selectedIndex: number
  onSelectedIndexChange: (index: number) => void
  onSelect: (chat: RecentChat) => void
  onHoverIndex?: (index: number) => void
  onClose: () => void
}

export const RecentChatsOverlay = memo(function RecentChatsOverlay({
  chats,
  selectedIndex,
  onSelectedIndexChange,
  onSelect,
  onHoverIndex,
  onClose,
}: RecentChatsOverlayProps) {
  const theme = useTheme()
  const [closeHover, setCloseHover] = useState(false)

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === 'escape') {
      key.preventDefault()
      onClose()
      return
    }

    if (chats.length === 0) return

    const plain = !key.ctrl && !key.meta && !key.option
    if (key.name === 'up' && plain) {
      key.preventDefault()
      onSelectedIndexChange(Math.max(0, selectedIndex - 1))
      return
    }
    if (key.name === 'down' && plain) {
      key.preventDefault()
      onSelectedIndexChange(Math.min(chats.length - 1, selectedIndex + 1))
      return
    }
    if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
      key.preventDefault()
      const chat = chats[selectedIndex]
      if (chat) onSelect(chat)
    }
  }, [onClose, chats, selectedIndex, onSelectedIndexChange, onSelect]))

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      <box style={{
        flexDirection: 'row',
        paddingLeft: 2,
        paddingRight: 2,
        paddingTop: 1,
        paddingBottom: 1,
        flexShrink: 0,
      }}>
        <text style={{ fg: theme.primary, flexGrow: 1 }}>
          <span attributes={TextAttributes.BOLD}>Recent Conversations</span>
        </text>
        <box style={{ flexDirection: 'row' }}>
          <Button
            onClick={onClose}
            onMouseOver={() => setCloseHover(true)}
            onMouseOut={() => setCloseHover(false)}
          >
            <text style={{ fg: closeHover ? theme.foreground : theme.muted }} attributes={TextAttributes.UNDERLINE}>Close</text>
          </Button>
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.DIM}>{' '}(Esc or Ctrl+R)  |  Arrow keys to navigate  |  Enter to select</span>
          </text>
        </box>
      </box>

      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>
          {'─'.repeat(80)}
        </text>
      </box>

      <scrollbox
        scrollX={false}
        scrollbarOptions={{ visible: false }}
        verticalScrollbarOptions={{
          visible: true,
          trackOptions: { width: 1 },
        }}
        style={{
          flexGrow: 1,
          rootOptions: {
            flexGrow: 1,
            backgroundColor: 'transparent',
          },
          wrapperOptions: {
            border: false,
            backgroundColor: 'transparent',
          },
          contentOptions: {
            paddingLeft: 1,
            paddingRight: 1,
            paddingTop: 1,
          },
        }}
      >
        {chats.length === 0 ? (
          <box style={{ paddingLeft: 1 }}>
            <text style={{ fg: theme.muted }}>No recent conversations found.</text>
          </box>
        ) : (
          chats.map((chat, index) => (
            <RecentChatEntry
              key={chat.id}
              chat={chat}
              isSelected={index === selectedIndex}
              onSelect={onSelect}
              onHover={() => onHoverIndex?.(index)}
            />
          ))
        )}
      </scrollbox>

      <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>
            {chats.length} conversation{chats.length === 1 ? '' : 's'}
          </span>
        </text>
      </box>
    </box>
  )
})
