import { memo, useState, useCallback, useMemo, useRef } from 'react'
import { TextAttributes, type KeyEvent, type ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { AtomRef } from '@effect-atom/atom-react'
import { Option } from 'effect'
import { useTheme } from '../../hooks/use-theme'
import { useAnimationStep } from '../../hooks/use-animation-time'

import { Button } from '../../components/button'
import { RecentChatEntry } from './recent-chat-entry'
import type { RecentChat } from '@magnitudedev/client-common'

interface RecentChatsOverlayProps {
  onClose: () => void
  onSelect: (chat: RecentChat) => void
  chats: RecentChat[]
  hasMore: boolean
  isLoading: boolean
  error: string | null
  loadMore: () => void
}

/** Threshold in pixels from the bottom to trigger loading more items. */
const LOAD_MORE_THRESHOLD = 3

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
  const [selectedIndex, setSelectedIndex] = useState(0)
  const scrollboxAtomRef = useMemo(
    () => AtomRef.make<Option.Option<ScrollBoxRenderable>>(Option.none()),
    [],
  )
  const scrolledSinceLoad = useRef(false)
  const lastScrollTop = useRef(0)

  // Infinite scroll: sample the shared animation clock every 160ms.
  // This is event-source-driven polling, not a reaction to state.
  const pollStep = useAnimationStep(hasMore, 160)
  const lastPollStepRef = useRef(pollStep)
  if (hasMore && pollStep !== lastPollStepRef.current) {
    lastPollStepRef.current = pollStep
    if (!isLoading) {
      const result = Option.match(scrollboxAtomRef.value, {
        onNone: () => false,
        onSome: (sb) => {
          const viewportHeight = sb.viewport?.height ?? 0
          const scrollTop = sb.scrollTop ?? 0
          const scrollHeight = sb.scrollHeight ?? 0

          if (scrollTop !== lastScrollTop.current) {
            scrolledSinceLoad.current = true
            lastScrollTop.current = scrollTop
          }

          if (scrollHeight <= viewportHeight) return false
          if (!scrolledSinceLoad.current) return false
          return scrollHeight - scrollTop - viewportHeight <= LOAD_MORE_THRESHOLD
        },
      })

      if (result) {
        loadMore()
      }
    }
  }

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
      setSelectedIndex(prev => Math.max(0, prev - 1))
      return
    }
    if (key.name === 'down' && plain) {
      key.preventDefault()
      setSelectedIndex(prev => Math.min(chats.length - 1, prev + 1))
      return
    }
    if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
      key.preventDefault()
      const chat = chats[selectedIndex]
      if (chat) onSelect(chat)
    }
  }, [onClose, chats, selectedIndex, onSelect]))

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
        ref={(sb: ScrollBoxRenderable | null) => { scrollboxAtomRef.set(Option.fromNullable(sb)) }}
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
                chat={chat}
                isSelected={index === selectedIndex}
                onSelect={onSelect}
                onHover={() => setSelectedIndex(index)}
              />
            ))}
            {/* Sentinel row: ensures scrollable overflow when hasMore is true
                so infinite scroll can detect near-bottom position.
                Also serves as a visual hint that more content exists below. */}
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
