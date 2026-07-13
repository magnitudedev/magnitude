import { memo, useState, useCallback, useRef } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import type { ActionId } from '../../types/ui-actions'
import type { ContextUsageDisplay, DisplayTimeline } from '@magnitudedev/sdk'
import type { TimelineStatus } from '@magnitudedev/client-common'
import { useTheme } from '../../hooks/use-theme'
import { useFilePanel } from '../../hooks/use-file-panel'
import { useLocalWidth } from '../../hooks/use-local-width'
import { Button } from '../../components/button'
import { ChatTimeline } from '../chat-timeline/timeline'
import { ContextUsageBar } from '../agent-status/context-usage-bar'

import { FileViewerPanel } from '../file-viewer/panel'
import { SelectedFileProvider } from '../../hooks/use-file-viewer'

interface ForkDetailOverlayProps {
  forkName: string
  forkRole: string
  timeline: DisplayTimeline | null
  timelineStatus: TimelineStatus['_tag']
  context: ContextUsageDisplay | null
  displayMode: 'default' | 'transcript'
  onClose: () => void
  onForkExpand?: (forkId: string) => void
  onErrorAction?: (actionId: ActionId) => void
  modelSummary: { role: string; model: string } | null
  contextHardCap: number | null
  cwd: string | null
  projectRoot: string
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

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
  forkName,
  forkRole,
  timeline,
  timelineStatus,
  context,
  displayMode,

  onClose,
  onForkExpand,
  onErrorAction,
  modelSummary,
  contextHardCap,
  cwd,
  projectRoot,
}: ForkDetailOverlayProps) {
  const theme = useTheme()
  const [closeHover, setCloseHover] = useState(false)

  const scrollboxRef = useRef<any>(null)
  const scrollboxWidth = useLocalWidth()

  useKeyboard(useCallback((key: KeyEvent) => {
    if (key.name === 'escape') {
      key.preventDefault?.()
      key.stopPropagation?.()
      onClose()
    }
  }, [onClose]))

  const {
    selectedFile,
    selectedFileContent,
    selectedFileStreaming,
    canRenderPanel,
    openFile,
    closeFilePanel,
  } = useFilePanel({
    cwd,
    toolState: null,
    projectRoot,
  })

  // On mount/open — snap to bottom after first paint/layout (ref-based, no useEffect)
  const snappedRef = useRef(false)
  if (!snappedRef.current) {
    snappedRef.current = true
    scheduleInitialForkOverlaySnap(
      () => scrollboxRef.current?.scrollTo(Number.MAX_SAFE_INTEGER),
    )
  }

  const tokenEstimate = context?.tokenEstimate ?? 0
  const tokenUsage = tokenEstimate > 0 ? tokenEstimate : null
  const isCompacting = context?.isCompacting ?? false

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
            ref={(el: any) => {
              scrollboxRef.current = el
              scrollboxWidth.ref.current = el
            }}
            onSizeChange={scrollboxWidth.onSizeChange}
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
            {timelineStatus === 'pending' ? (
              <box style={{ paddingLeft: 1 }}>
                <text style={{ fg: theme.muted }}>Loading activity...</text>
              </box>
            ) : timelineStatus === 'empty' || !timeline || timeline.presentation.entries.length === 0 ? (
              <box style={{ paddingLeft: 1 }}>
                <text style={{ fg: theme.muted }}>No activity yet.</text>
              </box>
            ) : (
              <ChatTimeline
                timeline={timeline}
                chatColumnWidth={scrollboxWidth.width ?? 80}
                themeErrorColor={theme.error}
                onFileClick={openFile}
                onForkExpand={onForkExpand ?? (() => {})}
                onErrorAction={onErrorAction ?? (() => {})}
              />
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
          {displayMode === 'transcript' && (
            <>
              <text style={{ fg: theme.info }}>Transcript Mode</text>
              <text style={{ fg: theme.muted }}>{' · '}</text>
            </>
          )}
          <text>
            <span fg={theme.muted}>{modelSummary?.role ?? '—'}</span>
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
