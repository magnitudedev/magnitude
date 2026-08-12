import { afterEach, describe, expect, test, vi } from 'vitest'
import React, { act, type ReactNode } from 'react'
import { create, type ReactTestRenderer } from 'react-test-renderer'
import type { InputValue } from '@magnitudedev/client-common'
import type { ComposerProps } from './types'
import { MultilineInput } from './multiline-input'
import { chatThemes } from '../../utils/theme'
import { PRIMARY_SLOT_ID } from '@magnitudedev/sdk'

const mountedRenderers: ReactTestRenderer[] = []

;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true

vi.mock('@opentui/react', async () => {
  const actual = await vi.importActual<typeof import('@opentui/react')>('@opentui/react')
  return {
    ...actual,
    useRenderer: () => ({ requestRender: () => {}, setMousePointer: () => {} }),
  }
})

vi.mock('@magnitudedev/client-common', async () => {
  const actual = await vi.importActual<typeof import('@magnitudedev/client-common')>('@magnitudedev/client-common')
  return {
    ...actual,
    useFileMentions: () => ({
      isOpen: false,
      query: '',
      items: [],
      recentItems: [],
      overflowCount: 0,
      selectedIndex: 0,
      confirmSelection: () => {},
      setSelectedIndex: () => {},
      handleKeyIntercept: () => false,
    }),
    useSlashCommands: () => ({
      isSlashMenuOpen: false,
      filteredCommands: [],
      selectedIndex: 0,
      setSelectedIndex: () => {},
      handleKeyIntercept: () => false,
      getSelectedCommandText: () => null,
    }),
    useAgentClient: () => ({
      query: () => ({ pipe: () => {} }),
      mutation: () => ({ pipe: () => {} }),
      runtime: { pipe: () => {} },
      pipe: () => {},
    }),
  }
})

vi.mock('@effect-atom/atom-react', async () => {
  const actual = await vi.importActual<typeof import('@effect-atom/atom-react')>('@effect-atom/atom-react')
  return {
    ...actual,
    useAtomValue: () => '',
    useAtomSet: () => () => {},
    useAtomMount: () => {},
  }
})

vi.mock('./chat-surface-keyboard', () => ({ ChatSurfaceKeyboard: () => null }))
vi.mock('./mention-menu', () => ({ FileMentionMenu: () => null }))
vi.mock('./slash-menu', () => ({ SlashCommandMenu: () => null }))
vi.mock('./attachment-bar', () => ({ AttachmentsBar: () => null }))
vi.mock('./context-usage', () => ({ ContextUsage: () => null, contextUsageWidth: () => 0 }))
vi.mock('./residency-indicator', () => ({ ResidencyIndicator: () => null }))
vi.mock('../../components/button', () => ({ Button: ({ children }: { children?: ReactNode }) => <>{children}</> }))
const { Composer } = await import('./composer')

const EMPTY_INPUT: InputValue = {
  text: '',
  cursorPosition: 0,
  lastEditDueToNav: false,
  pasteSegments: [],
  mentionSegments: [],
  selectedPasteSegmentId: null,
  selectedMentionSegmentId: null,
}

function makeProps(overrides: Partial<ComposerProps> = {}): ComposerProps {
  return {
    sessionId: null,
    cwd: null,
    clientWorkingDirectory: '/tmp/default',
      status: 'idle' as const,
      hasRunningForks: false,
      bashMode: false,
      modelsConfigured: true,
      modelSetupInProgress: false,
      modelSetupPlaceholder: null,
      modelSummary: { role: 'role', model: 'model', thinkingLevel: 'high' },
      localModels: null,
      modelSlots: null,
      selectedProviderId: null,
      selectedSlotId: PRIMARY_SLOT_ID,
      tokenUsage: null,
      contextHardCap: null,
      isCompacting: false,
      theme: chatThemes.dark,
      modeColor: '#00aaff',
      chatColumnWidth: 100,
      attachmentsMaxWidth: 80,
      composerCanFocus: false,
      widgetNavActive: false,
      isWorkerView: false,

      enableAutopilot: false,
      autopilotEnabled: false,
      autopilotGenerating: false,
      displayMode: 'default' as const,
      submitUserMessage: vi.fn(() => {}),
      runSlashCommand: vi.fn(() => false),
      executeBash: vi.fn((_command: string) => true),
      clearSystemBanners: vi.fn(() => {}),
      interruptFork: vi.fn(() => {}),
      interruptAll: vi.fn(() => {}),
      openSettings: vi.fn(() => {}),
      openHardware: vi.fn(() => {}),
      openCatalog: vi.fn(() => {}),
      thinkingOptions: [],
      applyThinking: vi.fn(() => {}),
      handleWidgetKeyEvent: vi.fn(() => false),
      enterBashMode: vi.fn(() => {}),
      exitBashMode: vi.fn(() => {}),
      showToast: vi.fn(() => {}),
      toggleAutopilot: vi.fn(() => {}),
    displayMessages: [],
    selectedForkId: null,
    isBlockingOverlayActive: false,
    selectedFileOpen: false,
    onCloseFilePanel: () => {},
    ...overrides,
  }
}

function multilineProps() {
  const renderer = mountedRenderers.at(-1)
  if (!renderer) throw new Error('Composer not mounted')
  return renderer.root.findByType(MultilineInput).props as {
    readonly value: string
    readonly onChange: (value: InputValue) => void
    readonly onSubmit: () => void
  }
}

function setComposerText(text: string) {
  const value: InputValue = {
    ...EMPTY_INPUT,
    text,
    cursorPosition: text.length,
  }
  act(() => {
    multilineProps().onChange(value)
  })
}

async function submitComposer() {
  await act(async () => {
    multilineProps().onSubmit()
    await Promise.resolve()
  })
}

async function mountComposer(props: ComposerProps) {
  let renderer: ReactTestRenderer | null = null
  await act(async () => {
    renderer = create(<Composer {...props} /> as React.ReactElement)
    await Promise.resolve()
  })
  mountedRenderers.push(renderer!)
  multilineProps()
}

afterEach(async () => {
  await act(async () => {
    for (const renderer of mountedRenderers.splice(0)) renderer.unmount()
    await Promise.resolve()
  })
  vi.clearAllMocks()
})

describe('ChatController submit slash behavior', () => {
  test('typed unknown slash text is routed and then sent as a normal message', async () => {
    const props = makeProps()
    await mountComposer(props)

    setComposerText('/foo')
    await submitComposer()

    expect(props.runSlashCommand).toHaveBeenCalledWith('/foo')
    expect(props.submitUserMessage).toHaveBeenCalledWith(expect.objectContaining({
      message: '/foo',
      visibleMessage: '/foo',
    }))

  })

  test('handled built-in slash text executes without sending', async () => {
    const props = makeProps({
      runSlashCommand: vi.fn((text: string) => text.trim() === '/new' || text.trim() === '/resume'),
    })
    await mountComposer(props)

    setComposerText('/new ')
    await submitComposer()
    setComposerText('/resume ')
    await submitComposer()

    expect(props.runSlashCommand).toHaveBeenNthCalledWith(1, '/new ')
    expect(props.runSlashCommand).toHaveBeenNthCalledWith(2, '/resume ')
    expect(props.submitUserMessage).not.toHaveBeenCalled()
    expect(multilineProps().value).toBe('')
  })

  test('unhandled slash-looking text remains a normal message', async () => {
    const props = makeProps()
    await mountComposer(props)

    setComposerText('/Users/me/a.png /Users/me/b.png')
    await submitComposer()
    setComposerText('/home/me/a.png')
    await submitComposer()

    expect(props.runSlashCommand).toHaveBeenCalledTimes(2)
    expect(props.submitUserMessage).toHaveBeenCalledWith(expect.objectContaining({ message: '/Users/me/a.png /Users/me/b.png' }))
    expect(props.submitUserMessage).toHaveBeenCalledWith(expect.objectContaining({ message: '/home/me/a.png' }))
  })
})
