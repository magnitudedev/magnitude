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
  nextCtrlCWillExit: boolean
  isSubagentView: boolean
}

export type ChatControllerServices = {
  submitUserMessage: (payload: {
    message: string
    visibleMessage?: string
    mentionAttachments?: Attachment[]
    attachments: Attachment[]
  }) => Promise<void> | void
  runSlashCommand: (commandText: string) => boolean | void
  executeBash: (command: string) => BashResult
  appendBashOutput: (result: BashResult) => void
  clearSystemBanners: () => void
  interrupt: () => void
  interruptAll: () => void
  openSettings: (tab: SettingsTab) => void
  toggleTaskPanel: () => void
  handleWidgetKeyEvent: (key: KeyEvent) => boolean
  enterBashMode: () => void
  exitBashMode: () => void
}

export type SubagentTabItem = {
  forkId: string
  agentId: string
  name: string
  startedAt: number
  toolCount: number
  toolSummaryLine: string
  statusLine: string
  phase: 'active' | 'exiting'
}

export type ChatControllerProps = {
  env: ChatControllerEnv
  services: ChatControllerServices
  displayMessages: DisplayState['messages']
  subagentTabs: SubagentTabItem[]
  selectedForkId: string | null
  onSubagentTabSelect: (forkId: string | null) => void
  selectedArtifactOpen: boolean
  onCloseArtifact: () => void
  onApprove: () => void
  onReject: () => void
  onInputHasTextChange?: (hasText: boolean) => void
  restoredQueuedInputText?: string | null
  onRestoredQueuedInputHandled?: () => void
}