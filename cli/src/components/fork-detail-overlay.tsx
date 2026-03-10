import { memo, useState, useEffect } from 'react'
import { TextAttributes } from '@opentui/core'
import type { DisplayState, DisplayMessage } from '@magnitudedev/agent'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { MessageView } from './message-view'
import { useCollapsedBlocks } from '../hooks/use-collapsed-blocks'

interface ForkDetailOverlayProps {
  forkId: string
  forkName: string
  forkRole: string
  initialPrompt?: string | null
  onClose: () => void
  onForkExpand?: (forkId: string) => void
  subscribeForkDisplay: (forkId: string, cb: (state: DisplayState) => void) => () => void
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export const ForkDetailOverlay = memo(function ForkDetailOverlay({
  forkId,
  forkName,
  forkRole,
  initialPrompt,
  onClose,
  onForkExpand,
  subscribeForkDisplay,
}: ForkDetailOverlayProps) {
  const theme = useTheme()
  const [closeHover, setCloseHover] = useState(false)
  const [display, setDisplay] = useState<DisplayState | null>(null)
  const [isPromptCollapsed, setIsPromptCollapsed] = useState(true)

  useEffect(() => setIsPromptCollapsed(true), [forkId, initialPrompt])
  const { isCollapsed, toggleCollapse } = useCollapsedBlocks()

  useEffect(() => {
    const unsubscribe = subscribeForkDisplay(forkId, (state) => {
      setDisplay(state)
    })
    return unsubscribe
  }, [forkId, subscribeForkDisplay])

  const messages = display?.messages ?? []
  const isStreaming = display?.status === 'streaming'
  const normalizedPrompt = initialPrompt?.trim() ?? ''
  const hasInitialPrompt = normalizedPrompt.length > 0
  const promptPreview = normalizedPrompt.length > 120 ? `${normalizedPrompt.slice(0, 120)}…` : normalizedPrompt

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

      {hasInitialPrompt && (
        <box
          style={{
            marginTop: 1,
            marginLeft: 2,
            marginRight: 2,
            marginBottom: 1,
            paddingLeft: 1,
            paddingRight: 1,
            paddingTop: 1,
            paddingBottom: 1,
            border: true,
            borderColor: theme.border,
            flexDirection: 'column',
            flexShrink: 0,
          }}
        >
          <box style={{ flexDirection: 'row' }}>
            <text style={{ flexGrow: 1 }}>
              <span fg={theme.muted} attributes={TextAttributes.BOLD}>Initial prompt</span>
            </text>
            <Button onClick={() => setIsPromptCollapsed(prev => !prev)}>
              <text style={{ fg: theme.info }}>
                {isPromptCollapsed ? 'Show' : 'Hide'}
              </text>
            </Button>
          </box>
          <box style={{ marginTop: 1 }}>
            <text style={{ fg: theme.muted }}>
              {isPromptCollapsed ? promptPreview : normalizedPrompt}
            </text>
          </box>
        </box>
      )}

      {/* Message list */}
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
              />
            )
          })
        )}
      </scrollbox>

      {/* Footer */}
      <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>
            {messages.filter(m => m.type === 'think_block').length} tool block{messages.filter(m => m.type === 'think_block').length === 1 ? '' : 's'}
          </span>
        </text>
      </box>
    </box>
  )
})
