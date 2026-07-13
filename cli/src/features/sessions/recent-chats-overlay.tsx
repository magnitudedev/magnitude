import { memo, useState, useCallback, useRef, useSyncExternalStore } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../../hooks/use-theme'
import { subscribeAnimationTick, getAnimationTickSnapshot } from '@magnitudedev/client-common'
import { safeRenderableAccess } from '../../utils/safe-renderable-access'
import { Button } from '../../components/button'
import { RecentChatEntry } from './recent-chat-entry'
import type { RecentChat, RecentChatsPage } from '@magnitudedev/client-common'

interface RecentChatsOverlayProps {
  onClose: () => void
  onSelect: (chat: RecentChat) => void
  loadPage: (offset: number, limit: number) => Promise<RecentChatsPage>
}

/** Approximate height of one chat entry row in terminal lines. */
const ROW_HEIGHT = 1
/** Threshold in pixels from the bottom to trigger loading more items. */
const LOAD_MORE_THRESHOLD = 3
/** Polling interval for scroll position checks (ms). */
const SCROLL_POLL_INTERVAL = 150
/** Minimum page size if viewport measurement isn't available yet. */
const MIN_PAGE_SIZE = 10

export const RecentChatsOverlay = memo(function RecentChatsOverlay({
  onClose,
  onSelect,
  loadPage,
}: RecentChatsOverlayProps) {
  const theme = useTheme()
  const [closeHover, setCloseHover] = useState(false)
  const scrollboxRef = useRef<any>(null)

  // Overlay-owned pagination state
  const [chats, setChats] = useState<RecentChat[]>([])
  const [hasMore, setHasMore] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [loadError, setLoadError] = useState<string | null>(null)
  const pageSizeRef = useRef(MIN_PAGE_SIZE)
  const loadingRef = useRef(false)
  const chatsRef = useRef<RecentChat[]>([])
  const measuredRef = useRef(false)
  const scrolledSinceLoad = useRef(false)
  const lastScrollTop = useRef(0)
  const initializedRef = useRef(false)

  // Keep chatsRef in sync (imperative, no useEffect)
  chatsRef.current = chats

  // Measure viewport and compute page size after mount
  // Returns true if measurement succeeded
  const measurePageSize = useCallback((): boolean => {
    return safeRenderableAccess(
      scrollboxRef.current,
      (sb) => {
        const viewportHeight = sb.viewport?.height ?? 0
        if (viewportHeight > 0) {
          pageSizeRef.current = Math.max(MIN_PAGE_SIZE, Math.floor(viewportHeight / ROW_HEIGHT))
          measuredRef.current = true
          return true
        }
        return false
      },
      { fallback: false },
    ) ?? false
  }, [])

  // Internal load function — appends items starting at given offset
  const doLoad = useCallback((offset: number, replace: boolean) => {
    if (loadingRef.current) return
    loadingRef.current = true
    setIsLoading(true)
    setLoadError(null)

    loadPage(offset, pageSizeRef.current)
      .then((page) => {
        if (replace) {
          setChats(page.items)
          chatsRef.current = page.items
        } else {
          setChats(prev => {
            const next = [...prev, ...page.items]
            chatsRef.current = next
            return next
          })
        }
        setHasMore(page.hasMore)
      })
      .catch((err) => {
        setLoadError(err instanceof Error ? err.message : 'Failed to load conversations')
      })
      .finally(() => {
        loadingRef.current = false
        setIsLoading(false)
        // Reset scroll gate — user must scroll again before next load triggers
        scrolledSinceLoad.current = false
      })
  }, [loadPage])

  // On mount: measure viewport, then load first page (ref-based, no useEffect)
  if (!initializedRef.current) {
    initializedRef.current = true
    setTimeout(() => {
      measurePageSize()
      doLoad(0, true)
    }, 0)
  }

  // Infinite scroll: use animation tick for polling (150ms ≈ 2 ticks)
  const tick = useSyncExternalStore(subscribeAnimationTick, getAnimationTickSnapshot, getAnimationTickSnapshot)
  // Check scroll position every ~2 ticks (160ms)
  const lastPollTickRef = useRef(0)
  if (hasMore && tick !== lastPollTickRef.current && tick % 2 === 0) {
    lastPollTickRef.current = tick
    if (!loadingRef.current) {
      const result = safeRenderableAccess(
        scrollboxRef.current,
        (sb) => {
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
        { fallback: false },
      )

      if (result) {
        doLoad(chatsRef.current.length, false)
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
        ref={scrollboxRef}
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
        {chats.length === 0 && !isLoading ? (
          <box style={{ paddingLeft: 1 }}>
            <text style={{ fg: theme.muted }}>No recent conversations found.</text>
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
                <text style={{ fg: theme.muted }}>
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
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>
            {isLoading ? 'Loading...' : loadError ? `Error: ${loadError}` : `${chats.length} conversation${chats.length === 1 ? '' : 's'}`}
          </span>
        </text>
      </box>
    </box>
  )
})
