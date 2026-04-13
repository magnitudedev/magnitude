import { beforeAll, expect, test, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import type { BashResult } from '../../utils/bash-executor'
import type { ChatTheme } from '../../types/theme-system'
import { initThemeStore } from '../../hooks/use-theme'
import type { ChatControllerProps } from './types'
import type { TaskListItem } from './task-list/index'

vi.mock('@opentui/react', async () => {
  const actual = await vi.importActual<typeof import('@opentui/react')>('@opentui/react')
  return {
    ...actual,
    useRenderer: () => ({
      requestRender: () => {},
    }),
  }
})

vi.mock('../../hooks/use-file-mentions', () => ({
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

vi.mock('../../hooks/use-slash-commands', () => ({
  useSlashCommands: () => ({
    isSlashMenuOpen: false,
    filteredCommands: [],
    selectedIndex: 0,
    setSelectedIndex: () => {},
    handleKeyIntercept: () => false,
  }),
}))

vi.mock('../chat-surface-keyboard', () => ({
  ChatSurfaceKeyboard: () => null,
}))

vi.mock('../file-mention-menu', () => ({
  FileMentionMenu: () => null,
}))

vi.mock('../slash-command-menu', () => ({
  SlashCommandMenu: () => null,
}))

vi.mock('../attachments-bar', () => ({
  AttachmentsBar: () => null,
}))

vi.mock('../context-usage-bar', () => ({
  ContextUsageBar: () => null,
}))

vi.mock('../button', () => ({
  Button: ({ children }: { children?: ReactNode }) => <>{children}</>,
}))

vi.mock('../multiline-input', () => ({
  INPUT_CURSOR_CHAR: '▍',
  MultilineInput: () => <text>[composer]</text>,
}))

vi.mock('./task-list', () => ({
  TaskList: () => <box>[task-list]</box>,
}))

const { ChatController } = await import('./chat-controller')

beforeAll(() => {
  initThemeStore()
})

const noop = () => {}

const theme: ChatTheme = {
  name: 'dark',
  primary: '#55aaff',
  secondary: '#ffaa00',
  success: '#00ff00',
  error: '#ff3333',
  warning: '#ffaa00',
  info: '#00aaff',
  link: '#55aaff',
  directory: '#55aaff',
  foreground: '#ffffff',
  background: '#000000',
  muted: '#888888',
  border: '#444444',
  surface: '#222222',
  surfaceHover: '#2a2a2a',
  aiLine: '#55aaff',
  userLine: '#ffaa00',
  userMessageBg: '#111111',
  userMessageHoverBg: '#1a1a1a',
  inputBg: '#111111',
  agentToggleExpandedBg: '#1a1a1a',
  agentFocusedBg: '#1a1a1a',
  agentContentBg: '#111111',
  terminalBg: '#000000',
  inputFg: '#cccccc',
  inputFocusedFg: '#ffffff',
  modeDefault: '#00aaff',
  modePlan: '#ffaa00',
  imageCardBorder: '#444444',
  syntax: {
    keyword: '#c084fc',
    string: '#86efac',
    number: '#93c5fd',
    comment: '#64748b',
    function: '#60a5fa',
    variable: '#e2e8f0',
    type: '#86efac',
    operator: '#94a3b8',
    property: '#e2e8f0',
    punctuation: '#64748b',
    literal: '#93c5fd',
    default: '#f1f5f9',
  },
}

function render(node: ReactNode) {
  return renderToStaticMarkup(<>{node}</>)
}

function makeTask(): TaskListItem {
  return {
    kind: 'task',
    rowId: 't-1',
    taskId: 't-1',
    title: 'Task 1',
    taskType: 'implement',
    status: 'pending',
    parentId: null,
    depth: 0,
    updatedAt: 0,
    assignee: {
      kind: 'worker',
      variant: 'idle',
      label: '[builder] builder-1',
      icon: '●',
      tone: 'muted',
      interactiveForkId: 'fork-1',
      timer: { startedAt: 0, resumedAt: null },
      resumed: false,
      continuityKey: 'fork-1',
      ghostEligible: true,
    },
  }
}

function makeBashResult(): BashResult {
  return {
    id: 'bash-1',
    command: '',
    stdout: '',
    stderr: '',

    exitCode: 0,
    cwd: '/tmp',
    timestamp: 0,
  }
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
      tokenUsage: 0,
      contextHardCap: null,
      isCompacting: false,
      theme,
      modeColor: '#00aaff',
      attachmentsMaxWidth: 60,
      composerCanFocus: false,
      widgetNavActive: false,
      isSubagentView: false,
    },
    services: {
      submitUserMessageToFork: noop,
      runSlashCommand: () => false,
      executeBash: () => makeBashResult(),
      appendBashOutput: noop,
      recordBashCommand: noop,
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
    tasks: [makeTask()],
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

  expect(html).toContain('Assigned To')
  expect(html).toContain('background-color:#111111;padding-top:1px;padding-left:1px;padding-right:2px')

  const taskListIndex = html.indexOf('Assigned To')
  const composerShellIndex = html.indexOf('background-color:#111111;padding-top:1px;padding-left:1px;padding-right:2px')
  expect(taskListIndex).toBeGreaterThan(-1)
  expect(composerShellIndex).toBeGreaterThan(taskListIndex)

  expect(html).not.toContain('vertical:╹')
  expect(html).not.toContain('horizontal:▀')
})