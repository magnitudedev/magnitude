import { memo, useState, useCallback, useMemo, useRef } from 'react'
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../../hooks/use-theme'
import { subscribeScrollboxActivity } from '../../utils/scroll-helpers'

import { Button } from '../../components/button'
import { RecentChatEntry } from './recent-chat-entry'
import {
  useInfiniteScroll,
  type RecentChat,
  type TimelineScrollAdapter,
} from '@magnitudedev/client-common'

interface RecentChatsOverlayProps {
  onClose: () => void
  onSelect: (chat: RecentChat) => void
  chats: RecentChat[]
  hasMore: boolean
  isLoading: boolean
  error: string | null
  loadMore: () => void
}

const recentChatOverlayRowId = (chatId: string): string =>
  `recent-chat-overlay:${chatId}`

export const RecentChatsOverlay = memo(function RecentChatsOverlay({
  onClose,
  onSelect,
  chats,
  hasMore,
  isLoading,
  error,
  loadMore,
}: RecentChatsOverlayProps) {
  const theme = useTheme()
  const [closeHover, setCloseHover] = useState(false)
  const [selectionIndex, setSelectionIndex] = useState(0)
  const scrollboxRef = useRef<ScrollBoxRenderable | null>(null)
  const selectedIndex = chats.length === 0
    ? -1
    : Math.min(selectionIndex, chats.length - 1)

  const scrollAdapter = useMemo<TimelineScrollAdapter>(
    () => ({
      getScrollMetrics: () => {
        const scrollbox = scrollboxRef.current
        if (scrollbox === null) return null
        return {
          scrollTop: scrollbox.scrollTop,
          viewportHeight: scrollbox.viewport.height,
          scrollHeight: scrollbox.scrollHeight,
        }
      },
      setScrollTop: (value) => {
        scrollboxRef.current?.scrollTo(Math.max(0, value))
      },
      subscribeActivity: (handler) => subscribeScrollboxActivity(scrollboxRef.current, handler),
      stickyThreshold: 2,
      loadThreshold: 3,
    }),
    [],
  )

  useInfiniteScroll({
    adapter: scrollAdapter,
    source: { hasMore, loadingMore: isLoading, loadMore },
    direction: 'bottom',
    fillViewport: true,
  })

  const moveSelection = useCallback((index: number) => {
    const chat = chats[index]
    if (!chat) return
    setSelectionIndex(index)
    scrollboxRef.current?.scrollChildIntoView(recentChatOverlayRowId(chat.id))
  }, [chats])

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
      moveSelection(Math.max(0, selectedIndex - 1))
      return
    }
    if (key.name === 'down' && plain) {
      key.preventDefault()
      moveSelection(Math.min(chats.length - 1, selectedIndex + 1))
      return
    }
    if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
      key.preventDefault()
      const chat = chats[selectedIndex]
      if (chat) onSelect(chat)
    }
  }, [onClose, chats, selectedIndex, onSelect, moveSelection]))

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
        <text style={{ fg: theme.accent, flexGrow: 1 }}>
          <span attributes={TextAttributes.BOLD}>Recent Conversations</span>
        </text>
        <box style={{ flexDirection: 'row' }}>
          <Button
            onClick={onClose}
            onMouseOver={() => setCloseHover(true)}
            onMouseOut={() => setCloseHover(false)}
          >
            <text style={{ fg: closeHover ? theme.text.body : theme.text.supporting }} attributes={TextAttributes.UNDERLINE}>Close</text>
          </Button>
          <text style={{ fg: theme.text.supporting }}>
            <span attributes={TextAttributes.DIM}>{' '}(Esc or Ctrl+R)  |  Arrow keys to navigate  |  Enter to select</span>
          </text>
        </box>
      </box>

      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border.standard }}>
          {'─'.repeat(80)}
        </text>
      </box>

      <scrollbox
        ref={(scrollbox: ScrollBoxRenderable | null) => { scrollboxRef.current = scrollbox }}
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
            backgroundColor: theme.background.canvas,
          },
          wrapperOptions: {
            border: false,
            backgroundColor: theme.background.canvas,
          },
          contentOptions: {
            paddingLeft: 1,
            paddingRight: 1,
            paddingTop: 1,
          },
        }}
      >
        {error ? (
          <box style={{ paddingLeft: 1 }}>
            <text style={{ fg: theme.status.failure }}>{error}</text>
          </box>
        ) : chats.length === 0 && !isLoading ? (
          <box style={{ paddingLeft: 1 }}>
            <text style={{ fg: theme.text.supporting }}>No recent conversations found.</text>
          </box>
        ) : (
          <>
            {chats.map((chat, index) => (
              <RecentChatEntry
                key={chat.id}
                id={recentChatOverlayRowId(chat.id)}
                chat={chat}
                isSelected={index === selectedIndex}
                onSelect={onSelect}
                onHover={() => setSelectionIndex(index)}
              />
            ))}
            {/* Status row for the next page and a visual hint that more content exists. */}
            {hasMore && (
              <box style={{ paddingTop: 1, paddingBottom: 1, paddingLeft: 1 }}>
                <text style={{ fg: theme.text.supporting }}>
                  <span attributes={TextAttributes.DIM}>
                    {isLoading ? '  Loading more...' : '  ↓ Scroll for more'}
                  </span>
                </text>
              </box>
            )}
          </>
        )}
      </scrollbox>

      <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.text.supporting }}>
          <span attributes={TextAttributes.DIM}>
            {isLoading
              ? 'Loading...'
              : error
                ? 'Unable to load conversations'
                : `${chats.length} conversation${chats.length === 1 ? '' : 's'}`}
          </span>
        </text>
      </box>
    </box>
  )
})
