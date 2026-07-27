import { expect, test, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { create, type ReactTestInstance } from 'react-test-renderer'
import { act, type ReactNode } from 'react'
import { Option } from 'effect'
import { TextAttributes } from '@opentui/core'
import type { ChatTheme } from '../../types/theme-system'
import type { ComposerProps } from './types'
import { PRIMARY_SLOT_ID, ReasoningEffortSchema, type TaskDisplayRow } from '@magnitudedev/sdk'
import {
  GIB,
  LOCAL_PROVIDER_ID,
  makeHardware,
  makeView,
  TEST_MEMORY_DOMAIN_ID,
} from '../local-inference/test-fixtures'

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true

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

vi.mock('./context-usage', () => ({
  ContextUsage: () => <text>[context]</text>,
  contextUsageWidth: () => '[context]'.length,
}))

vi.mock('./residency-indicator', () => ({
  ResidencyIndicator: () => <text>●</text>,
}))

vi.mock('../../components/button', () => ({
  Button: ({ children, ...props }: { children?: ReactNode; [key: string]: unknown }) => (
    <box {...props}>{children}</box>
  ),
}))

vi.mock('./multiline-input', () => ({
  INPUT_CURSOR_CHAR: '▍',
  MultilineInput: () => <text>[composer]</text>,
}))

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => theme,
}))

const { Composer, COMPOSER_BORDER_CHARS } = await import('./composer')

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
  menuBg: '#111111',
  menuAltBg: '#181818',
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
    clientWorkingDirectory: '/tmp/default',
    status: 'idle' as const,
    hasRunningForks: false,
    bashMode: false,
    modelsConfigured: true,
    modelSetupInProgress: false,
    modelSetupPlaceholder: null,
    modelSummary: { role: 'role', model: 'model', thinkingLevel: 'high' },
    localInferenceState: null,
    selectedProviderId: null,
    selectedSlotId: PRIMARY_SLOT_ID,
    tokenUsage: 5_000,
    contextHardCap: 100_000,
    isCompacting: false,
    displayMode: 'default' as const,
    theme,
    modeColor: '#00aaff',
    chatColumnWidth: 100,
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
    openHardware: noop,
    thinkingOptions: [],
    applyThinking: noop,
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
  const html = render(<Composer {...makeProps()} clientWorkingDirectory="/tmp/magnitude" />)

  expect(html).toContain('background-color:#111111;padding-top:1px;padding-bottom:1px;padding-left:1px;padding-right:2px')
  expect(html).toContain('border-style:single;border:left')
  expect(COMPOSER_BORDER_CHARS.vertical).toBe('┃')
  expect(html).toContain('>model<')
  expect(html).toContain('>high<')
  expect(html).toContain('style="fg:#c4b5fd"><span attributes="0">high</span>')
  expect(html).toContain('width:2px;flex-shrink:0')
  expect(html).toContain('/tmp/magnitude')
  expect(html).not.toContain('Thinking:')
  expect(html).not.toContain('Assigned To')

  expect(html).not.toContain('horizontal:▀')
  expect(html).not.toContain('horizontal:▄')
  expect(html).toContain('5k / 100k (5%)')
})

test('shows a single no-provider label instead of model and reasoning effort', () => {
  const html = render(<Composer {...makeProps()} modelsConfigured={false} />)

  expect(html).toContain('No provider configured')
  expect(html).not.toContain('>model<')
  expect(html).not.toContain('>high<')
})

test('shows resident memory three spaces after context and links it to hardware', () => {
  const openHardware = vi.fn()
  const localInferenceState = makeView({
    hardware: makeHardware({
      residentMemory: Option.some({
        domains: [{
          memoryDomainId: TEST_MEMORY_DOMAIN_ID,
          modelBytes: 13 * GIB,
          contextBytes: 2 * GIB,
          computeBytes: GIB,
          auxiliaryBytes: 0,
        }],
      }),
    }),
  })
  const html = render(
    <Composer
      {...makeProps()}
      localInferenceState={localInferenceState}
      selectedProviderId={LOCAL_PROVIDER_ID}
      openHardware={openHardware}
    />,
  )

  expect(html).toContain(
    '5k / 100k (5%)</text><box style="width:3px;flex-shrink:0"></box><box><text style="fg:#888888"><span attributes="0">16 GB mem',
  )

  let view!: ReturnType<typeof create>
  act(() => {
    view = create(
      <Composer
        {...makeProps()}
        localInferenceState={localInferenceState}
        selectedProviderId={LOCAL_PROVIDER_ID}
        openHardware={openHardware}
      />,
    )
  })
  const textOf = (node: ReactTestInstance): string => node.children
    .map((child) => typeof child === 'string' ? child : textOf(child))
    .join('')
  const memoryButton = view.root.findAll(
    (node) => typeof node.props.onClick === 'function',
  ).find((node) => textOf(node) === '16 GB mem')
  expect(memoryButton).toBeDefined()
  act(() => { (memoryButton!.props.onMouseOver as () => void)() })
  const hoveredMemory = view.root.findAll(
    (node) => node.type === 'text' && textOf(node) === '16 GB mem',
  )[0]!
  expect(hoveredMemory.props.style).toEqual({ fg: theme.primary })
  expect(hoveredMemory.findByType('span').props.attributes).toBe(TextAttributes.UNDERLINE)
  act(() => { (memoryButton!.props.onMouseOut as () => void)() })
  act(() => {
    (memoryButton!.props.onClick as () => void)()
  })
  expect(openHardware).toHaveBeenCalledOnce()
  act(() => { view.unmount() })
})

test('clicking effort opens the footer selector and clicking an option commits it', () => {
  const applied: string[] = []
  const localInferenceState = makeView({
    hardware: makeHardware({
      residentMemory: Option.some({
        domains: [{
          memoryDomainId: TEST_MEMORY_DOMAIN_ID,
          modelBytes: 16 * GIB,
          contextBytes: 0,
          computeBytes: 0,
          auxiliaryBytes: 0,
        }],
      }),
    }),
  })
  const thinkingOptions = ['none', 'low', 'medium', 'high'].map((value) => ({
    value: ReasoningEffortSchema.make(value),
    label: value.charAt(0).toUpperCase() + value.slice(1),
  }))
  let view!: ReturnType<typeof create>

  act(() => {
    view = create(
      <Composer
        {...makeProps()}
        localInferenceState={localInferenceState}
        selectedProviderId={LOCAL_PROVIDER_ID}
        thinkingOptions={thinkingOptions}
        applyThinking={(effort) => { applied.push(effort) }}
      />,
    )
  })

  const textOf = (node: ReactTestInstance): string => node.children
    .map((child) => typeof child === 'string' ? child : textOf(child))
    .join('')
  const clickable = () => view.root.findAll(
    (node) => typeof node.props.onClick === 'function',
  )

  const clickableLabels = clickable().map(textOf)
  expect(clickableLabels).toContain('high')
  expect(JSON.stringify(view.toJSON())).toContain('16 GB mem')
  const effort = clickable().find((node) => textOf(node) === 'high')
  expect(effort).toBeDefined()
  act(() => { (effort!.props.onClick as () => void)() })

  const openText = JSON.stringify(view.toJSON())
  expect(openText).toContain('None')
  expect(openText).toContain('Medium')
  expect(openText).toContain('Select reasoning level...')
  expect(openText).not.toContain('5k / 100k (5%)')
  expect(openText).not.toContain('16 GB mem')
  expect(openText).toContain('/tmp/default')

  const low = clickable().find((node) => textOf(node) === 'Low')
  expect(low).toBeDefined()
  act(() => { (low!.props.onClick as () => void)() })

  expect(applied).toEqual(['low'])
  expect(JSON.stringify(view.toJSON())).toContain('5k / 100k (5%)')
  expect(JSON.stringify(view.toJSON())).toContain('16 GB mem')
  act(() => { view.unmount() })
})
