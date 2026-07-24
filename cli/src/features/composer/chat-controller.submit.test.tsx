import { describe, expect, mock, test } from 'bun:test'
import React, { type ReactNode } from 'react'
import { act, create, type ReactTestRenderer } from 'react-test-renderer'
import type { InputValue } from '@magnitudedev/client-common'
import type { ComposerProps } from './types'
import { chatThemes } from '../../utils/theme'

let latestMultilineProps: {
  onChange: (value: InputValue) => void
  onSubmit: () => void
} | null = null

mock.module('../../hooks/use-file-mentions', () => ({
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
}))

mock.module('../../hooks/use-slash-commands', () => ({
  useSlashCommands: () => ({
    isSlashMenuOpen: false,
    filteredCommands: [],
    selectedIndex: 0,
    setSelectedIndex: () => {},
    handleKeyIntercept: () => false,
    getSelectedCommandText: () => null,
  }),
}))

mock.module('./chat-surface-keyboard', () => ({ ChatSurfaceKeyboard: () => null }))
mock.module('./mention-menu', () => ({ FileMentionMenu: () => null }))
mock.module('./slash-menu', () => ({ SlashCommandMenu: () => null }))
mock.module('./attachment-bar', () => ({ AttachmentsBar: () => null }))
mock.module('../agent-status/context-usage-bar', () => ({ ContextUsageBar: () => null }))
mock.module('../../components/button', () => ({ Button: ({ children }: { children?: ReactNode }) => <>{children}</> }))
mock.module('./multiline-input', () => ({
  INPUT_CURSOR_CHAR: '▍',
  MultilineInput: (props: { onChange: (value: InputValue) => void; onSubmit: () => void }) => {
    latestMultilineProps = { onChange: props.onChange, onSubmit: props.onSubmit }
    return <text>[composer]</text>
  },
}))

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
      status: 'idle' as const,
      hasRunningForks: false,
      bashMode: false,
      modelsConfigured: true,
      downloadSummary: null,
      modelSummary: { role: 'role', model: 'model', thinkingLevel: 'high' },
      tokenUsage: null,
      contextHardCap: null,
      isCompacting: false,
      theme: chatThemes.dark,
      modeColor: '#00aaff',
      attachmentsMaxWidth: 80,
      composerCanFocus: false,
      widgetNavActive: false,
      isWorkerView: false,

      enableAutopilot: false,
      autopilotEnabled: false,
      autopilotGenerating: false,
      displayMode: 'default' as const,
      submitUserMessage: mock(() => {}),
      runSlashCommand: mock(() => false),
      executeBash: mock((_command: string) => true),
      clearSystemBanners: mock(() => {}),
      interruptFork: mock(() => {}),
      interruptAll: mock(() => {}),
      openSettings: mock(() => {}),
      openCatalog: mock(() => {}),
      thinkingOptions: [],
      applyThinking: mock(() => {}),
      handleWidgetKeyEvent: mock(() => false),
      enterBashMode: mock(() => {}),
      exitBashMode: mock(() => {}),
      showToast: mock(() => {}),
      toggleAutopilot: mock(() => {}),
    displayMessages: [],
    selectedForkId: null,
    isBlockingOverlayActive: false,
    selectedFileOpen: false,
    onCloseFilePanel: () => {},
    ...overrides,
  }
}

function setComposerText(text: string) {
  if (!latestMultilineProps) throw new Error('MultilineInput not mounted')
  const value: InputValue = {
    ...EMPTY_INPUT,
    text,
    cursorPosition: text.length,
  }
  act(() => {
    latestMultilineProps!.onChange(value)
  })
}

async function submitComposer() {
  if (!latestMultilineProps) throw new Error('MultilineInput not mounted')
  await act(async () => {
    latestMultilineProps!.onSubmit()
    await Promise.resolve()
  })
}

describe('ChatController submit slash behavior', () => {
  test('typed unknown slash text sends as normal message and does not run slash command', async () => {
    let renderer: ReactTestRenderer | null = null
    const props = makeProps()
    act(() => {
      renderer = create(<Composer {...props} /> as React.ReactElement)
    })

    setComposerText('/foo')
    await submitComposer()

    expect(props.runSlashCommand).not.toHaveBeenCalled()
    expect(props.submitUserMessage).toHaveBeenCalledWith(expect.objectContaining({
      message: '/foo',
      visibleMessage: '/foo',
    }))

    act(() => renderer?.unmount())
  })

  test('slash-looking pasted text sends as normal message', async () => {
    let renderer: ReactTestRenderer | null = null
    const props = makeProps()
    act(() => {
      renderer = create(<Composer {...props} /> as React.ReactElement)
    })

    setComposerText('/new')
    await submitComposer()
    setComposerText('/Users/me/a.png /Users/me/b.png')
    await submitComposer()
    setComposerText('/home/me/a.png')
    await submitComposer()

    expect(props.runSlashCommand).not.toHaveBeenCalled()
    expect(props.submitUserMessage).toHaveBeenCalledWith(expect.objectContaining({ message: '/new' }))
    expect(props.submitUserMessage).toHaveBeenCalledWith(expect.objectContaining({ message: '/Users/me/a.png /Users/me/b.png' }))
    expect(props.submitUserMessage).toHaveBeenCalledWith(expect.objectContaining({ message: '/home/me/a.png' }))

    act(() => renderer?.unmount())
  })
})
