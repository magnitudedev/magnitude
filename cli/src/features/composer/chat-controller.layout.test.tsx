import { expect, test, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import { Option } from 'effect'
import type { ChatTheme } from '../../types/theme-system'
import type { ComposerProps } from './types'
import type { TaskDisplayRow } from '@magnitudedev/sdk'

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

vi.mock('./chat-surface-keyboard', () => ({
  ChatSurfaceKeyboard: () => null,
}))

vi.mock('./mention-menu', () => ({
  FileMentionMenu: () => null,
}))

vi.mock('./slash-menu', () => ({
  SlashCommandMenu: () => null,
}))

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

vi.mock('./attachment-bar', () => ({
  AttachmentsBar: () => null,
}))

vi.mock('../agent-status/context-usage-bar', () => ({
  ContextUsageBar: () => null,
}))

vi.mock('../../components/button', () => ({
  Button: ({ children }: { children?: ReactNode }) => <>{children}</>,
}))

vi.mock('./multiline-input', () => ({
  INPUT_CURSOR_CHAR: '▍',
  MultilineInput: () => <text>[composer]</text>,
}))

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => theme,
}))

const { Composer } = await import('./composer')

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
  diffGreenBg: '#122b22',
  diffRedBg: '#2c1919',
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

function makeTask(): TaskDisplayRow {
  return {
    kind: 'task',
    rowId: 't-1',
    taskId: 't-1',
    title: 'Task 1',

    status: 'pending',
    parentId: Option.none(),
    depth: 0,
    updatedAt: 0,
    assignee: {
      kind: 'actor',
      actorKey: 'fork-1',
      taskState: 'assigned',
      timer: Option.none(),
    },
  }
}

function makeProps(): ComposerProps {
  return {
    sessionId: null,
    cwd: null,
    status: 'idle' as const,
    hasRunningForks: false,
    bashMode: false,
    modelsConfigured: true,
    modelSummary: { role: 'role', model: 'model', thinkingLevel: 'high' },
    tokenUsage: 0,
    contextHardCap: null,
    isCompacting: false,
    displayMode: 'default' as const,
    theme,
    modeColor: '#00aaff',
    attachmentsMaxWidth: 60,
    composerCanFocus: false,
    widgetNavActive: false,
    isWorkerView: false,
    enableAutopilot: false,
    autopilotEnabled: false,
    autopilotGenerating: false,
    submitUserMessage: () => {},
    runSlashCommand: () => false,
    executeBash: () => true,
    clearSystemBanners: noop,
    interruptFork: noop,
    interruptAll: noop,
    openSettings: noop,
    handleWidgetKeyEvent: () => false,
    enterBashMode: noop,
    exitBashMode: noop,
    showToast: noop,
    toggleAutopilot: noop,
    displayMessages: [],
    selectedForkId: null,
    isBlockingOverlayActive: false,
    selectedFileOpen: false,
    onCloseFilePanel: noop,
  }
}

test('composer shell renders without an embedded task list (task list is the AgentStatus feature)', () => {
  const html = render(<Composer {...makeProps()} />)

  expect(html).toContain('background-color:#111111;padding-top:1px;padding-left:1px;padding-right:2px')
  expect(html).not.toContain('Assigned To')

  expect(html).not.toContain('vertical:╹')
  expect(html).not.toContain('horizontal:▀')
})

test('shows a single no-provider label instead of model and reasoning effort', () => {
  const html = render(<Composer {...makeProps()} modelsConfigured={false} />)

  expect(html).toContain('No provider configured')
  expect(html).not.toContain('>model<')
  expect(html).not.toContain('>high<')
})
