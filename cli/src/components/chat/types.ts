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

export type WorkerExecutionState = 'working' | 'idle' | 'killed'

export type WorkerExecutionSnapshot = {
  state: WorkerExecutionState
  activeSince: number | null
  accumulatedActiveMs: number
  completedAt: number | null
  resumeCount: number
}

export type TaskListItem = {
  taskId: string
  title: string
  type: string
  status: 'pending' | 'working' | 'completed' | 'archived'
  depth: number
  parentId: string | null
  createdAt: number
  updatedAt: number
  completedAt: number | null
  assignee: { kind: 'lead' } | { kind: 'none' } | { kind: 'user' } | { kind: 'worker'; workerType?: string; agentId: string }
  workerForkId: string | null
  workerExecution?: WorkerExecutionSnapshot | null
}

export type ChatControllerProps = {
  env: ChatControllerEnv
  services: ChatControllerServices
  displayMessages: DisplayState['messages']
  tasks: TaskListItem[]
  selectedForkId: string | null
  pushForkOverlay: (forkId: string) => void
  isBlockingOverlayActive: boolean
  selectedFileOpen: boolean
  onCloseFilePanel: () => void
  onApprove: () => void
  onReject: () => void
  onInputHasTextChange?: (hasText: boolean) => void
  restoredQueuedInputText?: string | null
  onRestoredQueuedInputHandled?: () => void
}