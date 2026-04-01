import type { Attachment, DisplayState } from '@magnitudedev/agent'
import type { BashResult } from '../../utils/bash-executor'
import type { KeyEvent } from '@opentui/core'
import type { ChatTheme } from '../../types/theme-system'
import type { SettingsTab } from '../../hooks/use-settings-navigation'

export type ChatControllerEnv = {
  status: DisplayState['status']
  pendingApproval: boolean
  hasRunningForks: boolean
  bashMode: boolean
  modelsConfigured: boolean
  modelSummary: { provider: string; model: string } | null
  tokenEstimate: number
  contextHardCap: number | null
  isCompacting: boolean
  theme: ChatTheme
  modeColor: string
  attachmentsMaxWidth: number
  composerCanFocus: boolean
  widgetNavActive: boolean
  isSubagentView: boolean
}

export type ChatControllerServices = {
  submitUserMessageToFork: (payload: {
    forkId: string | null
    message: string
    visibleMessage?: string
    mentionAttachments?: Attachment[]
    attachments: Attachment[]
  }) => Promise<void> | void
  runSlashCommand: (commandText: string) => boolean | void
  executeBash: (command: string) => BashResult | Promise<BashResult>
  appendBashOutput: (result: BashResult) => void
  clearSystemBanners: () => void
  interruptFork: (forkId: string | null) => void
  interruptAll: () => void
  openSettings: (tab: SettingsTab) => void
  handleWidgetKeyEvent: (key: KeyEvent) => boolean
  enterBashMode: () => void
  exitBashMode: () => void
  requestIdleSubagentClose: (payload: { forkId: string; agentId: string }) => void
  requestActiveSubagentKill: (payload: { forkId: string; agentId: string }) => void
}

export type SubagentTabItem = {
  forkId: string
  agentId: string
  role?: string
  name: string
  activeSince: number
  accumulatedActiveMs: number
  completedAt?: number
  resumeCount: number
  toolCount: number
  toolSummaryLine: string
  statusLine: string
  phase: 'active' | 'idle'
}

export type ChatControllerProps = {
  env: ChatControllerEnv
  services: ChatControllerServices
  displayMessages: DisplayState['messages']
  subagentTabs: SubagentTabItem[]
  selectedForkId: string | null
  pushForkOverlay: (forkId: string) => void
  selectedFileOpen: boolean
  onCloseFilePanel: () => void
  onApprove: () => void
  onReject: () => void
  onInputHasTextChange?: (hasText: boolean) => void
  restoredQueuedInputText?: string | null
  onRestoredQueuedInputHandled?: () => void
}