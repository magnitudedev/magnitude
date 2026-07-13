import { afterEach, beforeEach, expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactElement } from 'react'
import type { DisplayMessage, DisplayTimeline } from '@magnitudedev/sdk'

mock.module('../../hooks/use-theme', () => ({
  useTheme: () => ({
    muted: '#888888',
    primary: '#5e81ac',
    foreground: '#ffffff',
    border: '#4c566a',
  }),
}))

// useCollapsedBlocks was deleted — no mock needed

mock.module('../../components/button', () => ({
  Button: ({ children }: { children?: any }) => <>{children}</>,
}))

mock.module('../agent-status/context-usage-bar', () => ({
  ContextUsageBar: () => <text>[context]</text>,
}))

const openFileMock = mock(() => {})
const closeFilePanelMock = mock(() => {})
let filePanelState = {
  selectedFile: null as { path: string, section?: string } | null,
  selectedFileContent: null as string | null,
  selectedFileStreaming: null as any,
  canRenderPanel: false,
}

const selectedFileProviderCalls: Array<any> = []

let ForkDetailOverlay: typeof import('./fork-detail').ForkDetailOverlay

beforeEach(async () => {
  selectedFileProviderCalls.length = 0

  mock.module('../../hooks/use-file-viewer', () => ({
    SelectedFileProvider: ({ value, children }: { value: any, children?: any }) => {
      selectedFileProviderCalls.push(value)
      return <>{children}</>
    },
  }))

  mock.module('../../hooks/use-file-panel', () => ({
    useFilePanel: () => ({
      ...filePanelState,
      openFile: openFileMock,
      closeFilePanel: closeFilePanelMock,
    }),
  }))

  mock.module('../../hooks/use-local-width', () => ({
    useLocalWidth: () => ({
      ref: { current: null },
      onSizeChange: () => {},
      width: 80,
    }),
  }))

  mock.module('../chat-timeline/message-view', () => ({
    MessageView: ({ message, onFileClick }: { message: { id: string }, onFileClick?: (path: string, section?: string) => void }) => {
      if (message.id === 'm1') onFileClick?.('overlay.md', 'L1-L2')
      return <text>[message:{message.id}]</text>
    },
  }))

  // ChatTimeline is rendered inside the overlay; stub it so the test does not
  // pull in the full presentation pipeline (which would require the real
  // client-common barrel and OpenTUI render primitives).
  mock.module('../chat-timeline/timeline', () => ({
    ChatTimeline: ({ timeline, onFileClick }: { timeline: DisplayTimeline, onFileClick?: (path: string, section?: string) => void }) => {
      for (const entry of timeline.presentation.entries) {
        if (entry.kind === 'message') {
          const message = timeline.messages.byId[entry.messageId]
          if (message?.id === 'm1') onFileClick?.('overlay.md', 'L1-L2')
        }
      }
      return <text>[chat-timeline]</text>
    },
  }))

  mock.module('@magnitudedev/client-common', () => ({
    useAgentClient: () => null,
  }))

  mock.module('../file-viewer/panel', () => ({
    FileViewerPanel: ({ filePath }: { filePath: string }) => <text>[file-viewer:{filePath}]</text>,
  }))

  ;({ ForkDetailOverlay } = await import('./fork-detail'))
})

afterEach(() => {
  mock.restore()
})

const noop = () => {}

function makeTimeline(messages: DisplayMessage[]): DisplayTimeline {
  return {
    mode: 'idle',
    messages: {
      byId: Object.fromEntries(messages.map((m) => [m.id, m])),
      order: messages.map((m) => m.id),
    },
    streamingMessageId: null,
    window: {
      start: 0,
      end: messages.length,
      totalCount: messages.length,
      hasMoreBefore: false,
      hasMoreAfter: false,
    },
    presentation: {
      mode: 'default',
      entries: messages.map((message) => ({
        kind: 'message' as const,
        id: `message:${message.id}`,
        messageId: message.id,
        timestamp: message.timestamp,
        role: message.type === 'user_message' || message.type === 'queued_user_message' ? 'user' as const : 'assistant' as const,
        streaming: false,
        interrupted: false,
        nextMessageInterrupted: false,
      })),
      statusSlot: { kind: 'none' },
    },
  }
}

function propsWithTimeline(timeline: DisplayTimeline | null) {
  return {
    forkName: 'Fork One',
    forkRole: 'builder',
    timeline,
    timelineStatus: timeline && timeline.presentation.entries.length > 0 ? 'ready' as const : 'empty' as const,
    context: {
      tokenEstimate: 0,
      isCompacting: false,
    },
    displayMode: 'default' as const,
    onClose: noop,
    onForkExpand: noop,
    modelSummary: { role: 'role', model: 'model' },
    contextHardCap: null,
    client: null,
    sessionId: null,
    cwd: null,
    projectRoot: '/tmp',
  }
}

test('enables sticky bottom-follow semantics on the overlay scrollbox', async () => {
  filePanelState = { selectedFile: null, selectedFileContent: null, selectedFileStreaming: null, canRenderPanel: false }
  const html = renderToStaticMarkup(
    <ForkDetailOverlay
      {...propsWithTimeline(makeTimeline([]))}
    />,
  )

  const source = await Bun.file(new URL('./fork-detail.tsx', import.meta.url)).text()
  expect(source).toContain('stickyScroll')
  expect(html).toContain('stickyStart="bottom"')
})

test('mount snap helper is no longer exported', () => {
  expect(typeof (globalThis as Record<string, unknown>).scheduleInitialForkOverlaySnap).toBe('undefined')
})

test('routes overlay message file clicks to overlay-local openFile handler', async () => {
  openFileMock.mockClear()
  filePanelState = { selectedFile: null, selectedFileContent: null, selectedFileStreaming: null, canRenderPanel: false }

  renderToStaticMarkup(
    <ForkDetailOverlay
      {...propsWithTimeline(makeTimeline([{ id: 'm1', type: 'assistant_message', content: 'hello', timestamp: 1 }]))}
    /> as ReactElement,
  )

  expect(openFileMock).toHaveBeenCalledWith('overlay.md', 'L1-L2')
})

test('keeps overlay viewer rendering scoped to overlay-local file panel state', () => {
  filePanelState = { selectedFile: null, selectedFileContent: null, selectedFileStreaming: null, canRenderPanel: false }
  const closedHtml = renderToStaticMarkup(
    <ForkDetailOverlay
      {...propsWithTimeline(makeTimeline([]))}
    />,
  )
  expect(closedHtml).not.toContain('[file-viewer:')

  filePanelState = {
    selectedFile: { path: 'overlay.md' },
    selectedFileContent: '# Overlay',
    selectedFileStreaming: null,
    canRenderPanel: true,
  }
  const openHtml = renderToStaticMarkup(
    <ForkDetailOverlay
      {...propsWithTimeline(makeTimeline([]))}
    />,
  )
  expect(openHtml).toContain('[file-viewer:overlay.md]')
})



test('provides overlay-local selected file context from overlay file panel state', async () => {
  filePanelState = {
    selectedFile: { path: 'overlay.md' },
    selectedFileContent: '# Overlay',
    selectedFileStreaming: null,
    canRenderPanel: true,
  }

  renderToStaticMarkup(
    <ForkDetailOverlay
      {...propsWithTimeline(makeTimeline([{ id: 'm1', type: 'assistant_message', content: 'hello', timestamp: 1 }]))}
    /> as ReactElement,
  )

  expect(selectedFileProviderCalls.length).toBeGreaterThan(0)
  expect(selectedFileProviderCalls.at(-1)).toEqual({ path: 'overlay.md' })
})

test('overlay source has no message-update-driven imperative scroll-to-bottom effect', async () => {
  const source = await Bun.file(new URL('./fork-detail.tsx', import.meta.url)).text()

  // No useEffect should depend on [messages] — useMemo is fine for pure data transforms
  const useEffectDeps = [...source.matchAll(/useEffect\([^}]+\},\s*\[([^\]]+)\]\)/gs)].map(m => m[1])
  for (const deps of useEffectDeps) {
    expect(deps).not.toContain('messages')
  }
  expect(source).not.toContain('}, [display?.messages])')
  expect(source.match(/scrollTo\(Number\.MAX_SAFE_INTEGER\)/g)?.length ?? 0).toBe(1)
})
