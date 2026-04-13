import { describe, expect, mock, test } from 'bun:test'
import React, { type ReactNode } from 'react'
import { act, create, type ReactTestRenderer } from 'react-test-renderer'
import type { InputValue } from '../../types/store'
import type { ChatControllerProps } from './types'

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

mock.module('../chat-surface-keyboard', () => ({ ChatSurfaceKeyboard: () => null }))
mock.module('../file-mention-menu', () => ({ FileMentionMenu: () => null }))
mock.module('../slash-command-menu', () => ({ SlashCommandMenu: () => null }))
mock.module('../attachments-bar', () => ({ AttachmentsBar: () => null }))
mock.module('../context-usage-bar', () => ({ ContextUsageBar: () => null }))
mock.module('./task-list', () => ({ TaskList: () => null }))
mock.module('../button', () => ({ Button: ({ children }: { children?: ReactNode }) => <>{children}</> }))
mock.module('../multiline-input', () => ({
  INPUT_CURSOR_CHAR: '▍',
  MultilineInput: (props: { onChange: (value: InputValue) => void; onSubmit: () => void }) => {
    latestMultilineProps = { onChange: props.onChange, onSubmit: props.onSubmit }
    return <text>[composer]</text>
  },
}))

const { ChatController } = await import('./chat-controller')

const EMPTY_INPUT: InputValue = {
  text: '',
  cursorPosition: 0,
  lastEditDueToNav: false,
  pasteSegments: [],
  mentionSegments: [],
  selectedPasteSegmentId: null,
  selectedMentionSegmentId: null,
}

function makeProps(overrides: Partial<ChatControllerProps> = {}): ChatControllerProps {
  return {
    env: {
      status: 'idle',
      pendingApproval: false,
      hasRunningForks: false,
      bashMode: false,
      modelsConfigured: true,
      modelSummary: { provider: 'provider', model: 'model' },
      tokenUsage: null,
      contextHardCap: null,
      isCompacting: false,
      theme: {
        inputBg: '#111111',
        muted: '#888888',
        foreground: '#ffffff',
        primary: '#55aaff',
        secondary: '#ffaa00',
        border: '#444444',
        surface: '#222222',
        error: '#ff3333',
        inputFocusedFg: '#ffffff',
        inputFg: '#cccccc',
        info: '#00aaff',
      },
      modeColor: '#00aaff',
      attachmentsMaxWidth: 80,
      composerCanFocus: false,
      widgetNavActive: false,
      isSubagentView: false,
    },
    services: {
      submitUserMessageToFork: mock(() => {}),
      runSlashCommand: mock(() => false),
      executeBash: mock(() => ({ command: '', output: '', exitCode: 0 })),
      appendBashOutput: mock(() => {}),
      recordBashCommand: mock(() => {}),
      clearSystemBanners: mock(() => {}),
      interruptFork: mock(() => {}),
      interruptAll: mock(() => {}),
      openSettings: mock(() => {}),
      handleWidgetKeyEvent: mock(() => false),
      enterBashMode: mock(() => {}),
      exitBashMode: mock(() => {}),
      requestIdleSubagentClose: mock(() => {}),
      requestActiveSubagentKill: mock(() => {}),
    },
    displayMessages: [],
    tasks: [],
    selectedForkId: null,
    pushForkOverlay: () => {},
    isBlockingOverlayActive: false,
    selectedFileOpen: false,
    onCloseFilePanel: () => {},
    onApprove: () => {},
    onReject: () => {},
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
      renderer = create(<ChatController {...props} />)
    })

    setComposerText('/foo')
    await submitComposer()

    expect(props.services.runSlashCommand).not.toHaveBeenCalled()
    expect(props.services.submitUserMessageToFork).toHaveBeenCalledWith(expect.objectContaining({
      message: '/foo',
      visibleMessage: '/foo',
      forkId: null,
    }))

    act(() => renderer?.unmount())
  })

  test('slash-looking pasted text sends as normal message', async () => {
    let renderer: ReactTestRenderer | null = null
    const props = makeProps()
    act(() => {
      renderer = create(<ChatController {...props} />)
    })

    setComposerText('/new')
    await submitComposer()
    setComposerText('/Users/me/a.png /Users/me/b.png')
    await submitComposer()
    setComposerText('/home/me/a.png')
    await submitComposer()

    expect(props.services.runSlashCommand).not.toHaveBeenCalled()
    expect(props.services.submitUserMessageToFork).toHaveBeenCalledWith(expect.objectContaining({ message: '/new' }))
    expect(props.services.submitUserMessageToFork).toHaveBeenCalledWith(expect.objectContaining({ message: '/Users/me/a.png /Users/me/b.png' }))
    expect(props.services.submitUserMessageToFork).toHaveBeenCalledWith(expect.objectContaining({ message: '/home/me/a.png' }))

    act(() => renderer?.unmount())
  })
})
