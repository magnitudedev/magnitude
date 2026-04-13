import { memo, useState, useEffect, useCallback, useRef } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import type { CompactionState, DisplayState, DisplayMessage, ToolStateProjectionState } from '@magnitudedev/agent'
import { useTheme } from '../hooks/use-theme'
import { useFilePanel } from '../hooks/use-file-panel'
import { Button } from './button'
import { MessageView } from './message-view'
import { ContextUsageBar } from './context-usage-bar'
import { useCollapsedBlocks } from '../hooks/use-collapsed-blocks'
import { FileViewerPanel } from './file-viewer-panel'
import { SelectedFileProvider } from '../hooks/use-file-viewer'

interface ForkDetailOverlayProps {
  forkId: string
  forkName: string
  forkRole: string
  onClose: () => void
  onForkExpand?: (forkId: string) => void
  modelSummary: { provider: string; model: string } | null
  contextHardCap: number | null
  workspacePath: string | null
  projectRoot: string
  subscribeForkDisplay: (forkId: string, cb: (state: DisplayState) => void) => () => void
  subscribeForkCompaction: (forkId: string, cb: (state: CompactionState) => void) => () => void
  subscribeForkToolState: (forkId: string, cb: (state: ToolStateProjectionState) => void) => () => void
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

const EMPTY_MESSAGES: DisplayMessage[] = []

export function scheduleInitialForkOverlaySnap(
  scrollToBottom: () => void,
  schedule: typeof setTimeout = setTimeout,
  cancel: typeof clearTimeout = clearTimeout,
): () => void {
  const t1 = schedule(scrollToBottom, 0)
  const t2 = schedule(scrollToBottom, 50)
  return () => { cancel(t1); cancel(t2) }
}

export const ForkDetailOverlay = memo(function ForkDetailOverlay({
  forkId,
  forkName,
  forkRole,

  onClose,
  onForkExpand,
  modelSummary,
  contextHardCap,
  workspacePath,
  projectRoot,
  subscribeForkDisplay,
  subscribeForkCompaction,
  subscribeForkToolState,
}: ForkDetailOverlayProps) {
  const theme = useTheme()
  const [closeHover, setCloseHover] = useState(false)
  const [display, setDisplay] = useState<DisplayState | null>(null)
  const [toolState, setToolState] = useState<ToolStateProjectionState | null>(null)
  const [tokenEstimate, setTokenEstimate] = useState(0)
  const [lastActualInputTokens, setLastActualInputTokens] = useState<number | null>(null)
  const [hasCompletedTurn, setHasCompletedTurn] = useState(false)
  const [isCompacting, setIsCompacting] = useState(false)

  const scrollboxRef = useRef<any>(null)

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === 'escape') {
      key.preventDefault?.()
      key.stopPropagation?.()
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

  useEffect(() => {
    const unsubscribe = subscribeForkCompaction(forkId, (state) => {
      setTokenEstimate(state.tokenEstimate)
      setLastActualInputTokens(state.lastActualInputTokens)
      setHasCompletedTurn(state.hasCompletedTurn)
      setIsCompacting(state._tag !== 'idle')
    })
    return unsubscribe
  }, [forkId, subscribeForkCompaction])

  useEffect(() => {
    const unsubscribe = subscribeForkToolState(forkId, (state) => {
      setToolState(state)
    })
    return unsubscribe
  }, [forkId, subscribeForkToolState])

  const messages = display?.messages ?? EMPTY_MESSAGES
  const isStreaming = display?.status === 'streaming'

  const {
    selectedFile,
    selectedFileContent,
    selectedFileStreaming,
    canRenderPanel,
    openFile,
    closeFilePanel,
  } = useFilePanel({
    display,
    toolState,
    workspacePath,
    projectRoot,
  })

  // On mount/open — snap to bottom after first paint/layout
  useEffect(() => scheduleInitialForkOverlaySnap(
    () => scrollboxRef.current?.scrollTo(Number.MAX_SAFE_INTEGER),
  ), [])

  const tokenUsage = lastActualInputTokens ?? (hasCompletedTurn ? tokenEstimate : null)

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

      <SelectedFileProvider value={selectedFile}>
        <box style={{ flexDirection: 'row', flexGrow: 1, paddingLeft: 1, paddingRight: 1, gap: 1 }}>
          {/* Message list */}
          <scrollbox
            ref={scrollboxRef}
            stickyScroll
            stickyStart="bottom"
            scrollX={false}
            scrollbarOptions={{ visible: false }}
            verticalScrollbarOptions={{
              visible: true,
              trackOptions: { width: 1 },
            }}
            style={{
              width: canRenderPanel ? '60%' : '100%',
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
                    onFileClick={openFile}
                  />
                )
              })
            )}
          </scrollbox>

          {canRenderPanel && selectedFile && (
            <box style={{ width: '40%', minWidth: 36, height: '100%' }}>
              <FileViewerPanel
                filePath={selectedFile.path}
                content={selectedFileContent}
                scrollToSection={selectedFile.section}
                onClose={closeFilePanel}
                onOpenFile={openFile}
                streaming={selectedFileStreaming}
              />
            </box>
          )}
        </box>
      </SelectedFileProvider>

      <box style={{ flexShrink: 0, paddingTop: 1, paddingLeft: 2, paddingRight: 2, flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-end' }}>
        <box style={{ flexDirection: 'row', alignItems: 'center' }}>
          <text>
            <span fg={theme.muted}>{modelSummary?.provider ?? '—'}</span>
            <span fg={theme.muted}> {'\u00b7'} </span>
            <span fg={theme.foreground}>{modelSummary?.model ?? '—'}</span>
          </text>
          <box style={{ flexDirection: 'row', alignItems: 'center' }}>
            <text style={{ fg: theme.muted }}> | </text>
            <ContextUsageBar
              tokenUsage={tokenUsage}
              hardCap={contextHardCap}
              isCompacting={isCompacting}
            />
          </box>
        </box>
      </box>
    </box>
  )
})
