import type { DisplayMessage, DisplayTimeline, RawImageAttachment, RawMentionOccurrence } from '@magnitudedev/sdk'
import type { BashResult } from '@magnitudedev/client-common'
import type { KeyEvent } from '@opentui/core'
import type { ChatTheme } from '../../types/theme-system'

/**
 * Composer presentational props — individual and typed (spec §5.6: no prop
 * bags). All logic lives in the container via useComposerState; the composer
 * owns only draft text, cursor, segments, menus, and history navigation.
 */
export type ComposerProps = {
  sessionId: string | null
  cwd: string | null

  // Display-derived state
  status: DisplayTimeline['mode']
  hasRunningForks: boolean
  bashMode: boolean
  modelsConfigured: boolean
  modelSummary: { role: string; model: string; thinkingLevel: string } | null
  tokenUsage: number | null
  contextHardCap: number | null
  isCompacting: boolean
  displayMode: 'default' | 'transcript'

  // Presentation
  theme: ChatTheme
  modeColor: string
  attachmentsMaxWidth: number
  composerCanFocus: boolean
  widgetNavActive: boolean
  isWorkerView: boolean

  // Autopilot (disabled)
  enableAutopilot: boolean
  autopilotEnabled: boolean
  autopilotGenerating: boolean

  // Actions
  submitUserMessage: (payload: {
    message: string
    visibleMessage?: string
    imageAttachments: RawImageAttachment[]
    mentions: RawMentionOccurrence[]
  }) => void
  runSlashCommand: (commandText: string) => boolean | void
  executeBash: (command: string) => BashResult | Promise<BashResult | null> | null
  clearSystemBanners: () => void
  interruptFork: (forkId: string | null) => void
  interruptAll: () => void
  openSettings: () => void
  handleWidgetKeyEvent: (key: KeyEvent) => boolean
  enterBashMode: () => void
  exitBashMode: () => void
  showToast: (message: string) => void
  toggleAutopilot: () => void

  // Timeline context the input needs (queued-message display, history seed)
  displayMessages: readonly DisplayMessage[]
  selectedForkId: string | null

  // Layout/overlay coordination
  isBlockingOverlayActive: boolean
  selectedFileOpen: boolean
  onCloseFilePanel: () => void
}
