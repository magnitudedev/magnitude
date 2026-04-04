import { afterEach, beforeEach, expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { act, create } from 'react-test-renderer'
import type { CompactionState, DisplayState } from '@magnitudedev/agent'

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    muted: '#888888',
    primary: '#5e81ac',
    foreground: '#ffffff',
    border: '#4c566a',
  }),
}))

mock.module('../hooks/use-collapsed-blocks', () => ({
  useCollapsedBlocks: () => ({
    isCollapsed: () => false,
    toggleCollapse: () => {},
  }),
}))

mock.module('./button', () => ({
  Button: ({ children }: { children?: any }) => <>{children}</>,
}))

const openFileMock = mock(() => {})
const closeFilePanelMock = mock(() => {})
let filePanelState = {
  selectedFile: null as { path: string, section?: string } | null,
  selectedFileContent: null as string | null,
  selectedFileStreaming: null as any,
  canRenderPanel: false,
}

let ForkDetailOverlay: typeof import('./fork-detail-overlay').ForkDetailOverlay
let scheduleInitialForkOverlaySnap: typeof import('./fork-detail-overlay').scheduleInitialForkOverlaySnap

beforeEach(async () => {
  mock.module('../hooks/use-file-panel', () => ({
    useFilePanel: () => ({
      ...filePanelState,
      openFile: openFileMock,
      closeFilePanel: closeFilePanelMock,
    }),
  }))

  mock.module('./message-view', () => ({
    MessageView: ({ message, onFileClick }: { message: { id: string }, onFileClick?: (path: string, section?: string) => void }) => {
      if (message.id === 'm1') onFileClick?.('overlay.md', 'L1-L2')
      return <text>[message:{message.id}]</text>
    },
  }))

  mock.module('./file-viewer-panel', () => ({
    FileViewerPanel: ({ filePath }: { filePath: string }) => <text>[file-viewer:{filePath}]</text>,
  }))

  ;({ ForkDetailOverlay, scheduleInitialForkOverlaySnap } = await import('./fork-detail-overlay'))
})

afterEach(() => {
  mock.restore()
})

const noop = () => {}
const idleCompaction: CompactionState = { _tag: 'idle', tokenEstimate: 0 }

function makeDisplayState(messages: DisplayState['messages']): DisplayState {
  return {
    status: 'idle',
    messages,
    pendingInboundCommunications: [],
  } as unknown as DisplayState
}

function propsWithDisplay(display: DisplayState) {
  return {
    forkId: 'fork-1',
    forkName: 'Fork One',
    forkRole: 'builder',
    onClose: noop,
    onForkExpand: noop,
    modelSummary: { provider: 'provider', model: 'model' },
    contextHardCap: null,
    workspacePath: '/tmp',
    projectRoot: '/tmp',
    subscribeForkDisplay: (_forkId: string, cb: (state: DisplayState) => void) => {
      cb(display)
      return noop
    },
    subscribeForkCompaction: (_forkId: string, cb: (state: CompactionState) => void) => {
      cb(idleCompaction)
      return noop
    },
  }
}

test('enables sticky bottom-follow semantics on the overlay scrollbox', async () => {
  filePanelState = { selectedFile: null, selectedFileContent: null, selectedFileStreaming: null, canRenderPanel: false }
  const html = renderToStaticMarkup(
    <ForkDetailOverlay
      {...propsWithDisplay(makeDisplayState([]))}
    />,
  )

  const source = await Bun.file(new URL('./fork-detail-overlay.tsx', import.meta.url)).text()
  expect(source).toContain('stickyScroll')
  expect(html).toContain('stickyStart="bottom"')
})

test('mount snap helper schedules immediate + deferred bottom snaps and executes both', () => {
  const scheduled: Array<{ fn: () => void, delay: number, id: number }> = []
  const canceled: number[] = []
  let nextId = 1
  let snapCount = 0

  const cleanup = scheduleInitialForkOverlaySnap(
    () => { snapCount += 1 },
    ((fn: (...args: any[]) => void, delay?: number) => {
      const id = nextId++
      scheduled.push({ fn: () => fn(), delay: delay ?? 0, id })
      return id as unknown as ReturnType<typeof setTimeout>
    }) as typeof setTimeout,
    ((id: ReturnType<typeof setTimeout>) => {
      canceled.push(id as unknown as number)
    }) as typeof clearTimeout,
  )

  expect(scheduled.map((t) => t.delay)).toEqual([0, 50])

  scheduled[0]?.fn()
  scheduled[1]?.fn()
  expect(snapCount).toBe(2)

  cleanup()
  expect(canceled).toEqual([scheduled[0]!.id, scheduled[1]!.id])
})

test('routes overlay message file clicks to overlay-local openFile handler', async () => {
  openFileMock.mockClear()
  filePanelState = { selectedFile: null, selectedFileContent: null, selectedFileStreaming: null, canRenderPanel: false }

  await act(async () => {
    create(
      <ForkDetailOverlay
        {...propsWithDisplay(makeDisplayState([{ id: 'm1', type: 'assistant_message' } as any]))}
      />,
    )
  })

  expect(openFileMock).toHaveBeenCalledWith('overlay.md', 'L1-L2')
})

test('keeps overlay viewer rendering scoped to overlay-local file panel state', () => {
  filePanelState = { selectedFile: null, selectedFileContent: null, selectedFileStreaming: null, canRenderPanel: false }
  const closedHtml = renderToStaticMarkup(
    <ForkDetailOverlay
      {...propsWithDisplay(makeDisplayState([]))}
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
      {...propsWithDisplay(makeDisplayState([]))}
    />,
  )
  expect(openHtml).toContain('[file-viewer:overlay.md]')
})



test('overlay source has no message-update-driven imperative scroll-to-bottom effect', async () => {
  const source = await Bun.file(new URL('./fork-detail-overlay.tsx', import.meta.url)).text()

  expect(source).not.toContain('}, [messages])')
  expect(source).not.toContain('}, [display?.messages])')
  expect(source.match(/scrollTo\(Number\.MAX_SAFE_INTEGER\)/g)?.length ?? 0).toBe(1)
})