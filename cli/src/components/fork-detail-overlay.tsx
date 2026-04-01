import { memo, useState, useEffect, useCallback, useRef } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import type { DisplayState, DisplayMessage } from '@magnitudedev/agent'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { MessageView } from './message-view'
import { useCollapsedBlocks } from '../hooks/use-collapsed-blocks'

interface ForkDetailOverlayProps {
  forkId: string
  forkName: string
  forkRole: string
  onClose: () => void
  onForkExpand?: (forkId: string) => void
  onFileClick?: (path: string, section?: string) => void
  subscribeForkDisplay: (forkId: string, cb: (state: DisplayState) => void) => () => void
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

const EMPTY_MESSAGES: DisplayMessage[] = []

export const ForkDetailOverlay = memo(function ForkDetailOverlay({
  forkId,
  forkName,
  forkRole,

  onClose,
  onForkExpand,
  onFileClick,
  subscribeForkDisplay,
}: ForkDetailOverlayProps) {
  const theme = useTheme()
  const [closeHover, setCloseHover] = useState(false)
  const [display, setDisplay] = useState<DisplayState | null>(null)

  const scrollboxRef = useRef<any>(null)

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === 'escape') {
      key.preventDefault()
      onClose()
    }
  }, [onClose]))

  const { isCollapsed, toggleCollapse } = useCollapsedBlocks()

  useEffect(() => {
    const unsubscribe = subscribeForkDisplay(forkId, (state) => {
      setDisplay(state)
    })
    return unsubscribe
  }, [forkId, subscribeForkDisplay])

  const messages = display?.messages ?? EMPTY_MESSAGES
  const isStreaming = display?.status === 'streaming'

  // On mount — snap to bottom immediately
  useEffect(() => {
    const t1 = setTimeout(() => scrollboxRef.current?.scrollTo(Number.MAX_SAFE_INTEGER), 0)
    const t2 = setTimeout(() => scrollboxRef.current?.scrollTo(Number.MAX_SAFE_INTEGER), 50)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  // On new messages — snap to bottom with double timeout
  useEffect(() => {
    const t1 = setTimeout(() => scrollboxRef.current?.scrollTo(Number.MAX_SAFE_INTEGER), 0)
    const t2 = setTimeout(() => scrollboxRef.current?.scrollTo(Number.MAX_SAFE_INTEGER), 50)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [messages])
  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <box style={{
        flexDirection: 'row',
        paddingLeft: 2,
        paddingRight: 2,
        paddingTop: 1,
        paddingBottom: 1,
        flexShrink: 0,
      }}>
        <text style={{ flexGrow: 1 }}>
          <span fg={theme.muted} attributes={TextAttributes.BOLD}>{capitalize(forkRole)}:</span>
          {' '}
          <span fg={theme.primary} attributes={TextAttributes.BOLD}>{forkName}</span>
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
            <span attributes={TextAttributes.DIM}>{' '}(Esc)</span>
          </text>
        </box>
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>
          {'─'.repeat(80)}
        </text>
      </box>

      {/* Message list */}
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
        {messages.length === 0 ? (
          <box style={{ paddingLeft: 1 }}>
            <text style={{ fg: theme.muted }}>No activity yet.</text>
          </box>
        ) : (
          messages.map((msg: DisplayMessage) => {
            const isStreamingMsg = isStreaming && msg === messages[messages.length - 1] && msg.type === 'assistant_message'
            return (
              <MessageView
                key={msg.id}
                message={msg}
                isStreaming={isStreamingMsg}
                isCollapsed={msg.type === 'think_block' ? isCollapsed(msg.id) : undefined}
                onToggleCollapse={msg.type === 'think_block' ? () => toggleCollapse(msg.id) : undefined}
                onForkExpand={onForkExpand}
                onFileClick={onFileClick}
              />
            )
          })
        )}
      </scrollbox>

    </box>
  )
})
