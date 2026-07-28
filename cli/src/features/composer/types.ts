import type { DisplayMessage, DisplayTimeline, LocalModelsState, ModelSlotsState, ProviderId, RawImageAttachment, RawMentionOccurrence, ReasoningEffort, SlotId } from '@magnitudedev/sdk'
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
  clientWorkingDirectory: string

  // Display-derived state
  status: DisplayTimeline['mode']
  hasRunningForks: boolean
  bashMode: boolean
  modelsConfigured: boolean
  modelSetupInProgress: boolean
  modelSetupPlaceholder: string | null
  modelSummary: { role: string; model: string; thinkingLevel: string } | null
  localModels: LocalModelsState | null
  modelSlots: ModelSlotsState | null
  selectedProviderId: ProviderId | null
  selectedSlotId: SlotId
  tokenUsage: number | null
  contextHardCap: number | null
  isCompacting: boolean
  displayMode: 'default' | 'transcript'

  // Presentation
  theme: ChatTheme
  modeColor: string
  chatColumnWidth: number
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
  executeBash: (command: string) => boolean | Promise<boolean>
  clearSystemBanners: () => void
  interruptFork: (forkId: string | null) => void
  interruptAll: () => void
  openSettings: () => void
  openHardware: () => void
  thinkingOptions: readonly { value: ReasoningEffort; label: string }[]
  applyThinking: (effort: ReasoningEffort) => void
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
