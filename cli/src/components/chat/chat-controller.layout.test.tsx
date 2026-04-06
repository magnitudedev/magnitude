import { expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import type { ChatControllerProps } from './types'

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
  }),
}))

mock.module('../chat-surface-keyboard', () => ({
  ChatSurfaceKeyboard: () => null,
}))

mock.module('../file-mention-menu', () => ({
  FileMentionMenu: () => null,
}))

mock.module('../slash-command-menu', () => ({
  SlashCommandMenu: () => null,
}))

mock.module('../attachments-bar', () => ({
  AttachmentsBar: () => null,
}))

mock.module('../context-usage-bar', () => ({
  ContextUsageBar: () => null,
}))

mock.module('../button', () => ({
  Button: ({ children }: { children?: ReactNode }) => <>{children}</>,
}))

mock.module('../multiline-input', () => ({
  INPUT_CURSOR_CHAR: '▍',
  MultilineInput: () => <text>[composer]</text>,
}))

mock.module('./task-list', () => ({
  TaskList: () => <box>[task-list]</box>,
}))

const { ChatController } = await import('./chat-controller')

const noop = () => {}

function render(node: ReactNode) {
  return renderToStaticMarkup(<>{node}</>)
}

function makeProps(): ChatControllerProps {
  return {
    env: {
      status: 'idle',
      pendingApproval: false,
      hasRunningForks: false,
      bashMode: false,
      modelsConfigured: true,
      modelSummary: { provider: 'provider', model: 'model' },
      contextTokens: 0,
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
      attachmentsMaxWidth: 60,
      composerCanFocus: false,
      widgetNavActive: false,
      isSubagentView: false,
    },
    services: {
      submitUserMessageToFork: noop,
      runSlashCommand: () => false,
      executeBash: () => ({ command: '', output: '', exitCode: 0 }),
      appendBashOutput: noop,
      clearSystemBanners: noop,
      interruptFork: noop,
      interruptAll: noop,
      openSettings: noop,
      handleWidgetKeyEvent: () => false,
      enterBashMode: noop,
      exitBashMode: noop,
      requestIdleSubagentClose: noop,
      requestActiveSubagentKill: noop,
    },
    displayMessages: [],
    tasks: [
      {
        forkId: 'fork-1',
        agentId: 'builder-1',
        role: 'builder',
        name: 'Task 1',
        phase: 'idle',
        activeSince: 0,
        completedAt: undefined,
        accumulatedActiveMs: 0,
        resumeCount: 0,
        statusLine: '',
        toolSummaryLine: '',
        toolCount: 0,
      },
    ],
    selectedForkId: null,
    pushForkOverlay: noop,
    isBlockingOverlayActive: false,
    selectedFileOpen: false,
    onCloseFilePanel: noop,
    onApprove: noop,
    onReject: noop,
    onInputHasTextChange: noop,
    restoredQueuedInputText: null,
    onRestoredQueuedInputHandled: noop,
  }
}

test('renders task list directly above composer shell while preserving normal composer spacing and no top-cap row', () => {
  const html = render(<ChatController {...makeProps()} />)

  expect(html).toContain('[task-list]')
  expect(html).toContain('background-color:#111111;padding-top:1px;padding-left:1px;padding-right:2px')

  const taskListIndex = html.indexOf('[task-list]')
  const composerShellIndex = html.indexOf('background-color:#111111;padding-top:1px;padding-left:1px;padding-right:2px')
  expect(taskListIndex).toBeGreaterThan(-1)
  expect(composerShellIndex).toBeGreaterThan(taskListIndex)

  expect(html).not.toContain('vertical:╹')
  expect(html).not.toContain('horizontal:▀')
})