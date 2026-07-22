/**
 * Composer feature container (spec §5.6) — all composer logic comes from the
 * shared useComposerState hook; this container adapts it to the terminal
 * presentational Composer with individual typed props. The CommandContext is
 * the CLI's slash-command surface (overlays, system messages, bash mode).
 */
import { useCallback, useMemo, useRef, type ReactNode } from 'react'
import { useAtomValue, useAtomSet } from '@effect-atom/atom-react'
import type { KeyEvent } from '@opentui/core'
import {
  useComposerState,
  useInterruptActions,
  useSessionActions,
  useSlotProfiles,
  useDisplayState,
  getFork,
  orderedMessages,
  addSystemMessage,
  clearSystemMessages,
  useSelectedSessionId,
  selectedCwdAtom,
  selectedFilePathAtom,
  settingsOpenAtom,
  usageOpenAtom,
  bashModeAtom,
  useDisplayViewController,
  type CommandContext,
} from '@magnitudedev/client-common'
import type { RawImageAttachment, RawMentionOccurrence } from '@magnitudedev/sdk'
import { addEphemeralMessage } from '@magnitudedev/client-common'
import { cloudModelsOpenAtom, showRecentChatsOverlayAtom } from '../../state/cli-atoms'
import { useTheme } from '../../hooks/use-theme'
import { INIT_PROMPT } from '../../commands/init-prompt'
import { Composer } from './composer'
import { allowProviderMessageSend } from './provider-send-guard'

export function ComposerContainer({
  chatColumnWidth,
  widgetNavActive,
  handleWidgetKeyEvent,
  modelsConfigured,
}: {
  chatColumnWidth: number
  widgetNavActive: boolean
  handleWidgetKeyEvent: (key: KeyEvent) => boolean
  modelsConfigured: boolean
}): ReactNode {
  const theme = useTheme()
  const sessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const setCloudModelsOpen = useAtomSet(cloudModelsOpenAtom)
  const setBashMode = useAtomSet(bashModeAtom)
  const setShowRecentChats = useAtomSet(showRecentChatsOverlayAtom)
  const { displayMode, expandedForkStack } = useDisplayViewController()
  const selectedFilePath = useAtomValue(selectedFilePathAtom)
  const setSelectedFilePath = useAtomSet(selectedFilePathAtom)
  const showRecentChats = useAtomValue(showRecentChatsOverlayAtom)
  const settingsOpen = useAtomValue(settingsOpenAtom)
  const usageOpen = useAtomValue(usageOpenAtom)
  const { startNewSession } = useSessionActions()

  const showErrorToast = useCallback((message: string) => {
    addEphemeralMessage(message, theme.error)
  }, [theme.error])

  // Slash commands may trigger a send (skills, /init) — the hook that owns
  // sending is constructed with this context, so route through a ref.
  const sendRef = useRef<(text: string) => void>(() => {})

  const commandContext: CommandContext = useMemo(() => ({
    resetConversation: () => {
      startNewSession({ cwd: process.cwd() })
    },
    showSystemMessage: (message: string) => { addSystemMessage(message) },
    exitApp: () => { process.kill(process.pid, 'SIGINT') },
    openRecentChats: () => setShowRecentChats(true),
    enterBashMode: () => setBashMode(true),
    activateSkill: (skillName: string, _skillPath: string | undefined, args: string) => {
      const content = args.trim() ? `/${skillName} ${args.trim()}` : `/${skillName}`
      sendRef.current(content)
    },
    initProject: () => { void sendRef.current(INIT_PROMPT) },
    openSettings: () => setSettingsOpen(true),
    openUsage: () => setUsageOpen(true),
    openCloud: () => setCloudModelsOpen(true),
    toggleAutopilot: () => { /* disabled */ },
  }), [startNewSession, setShowRecentChats, setBashMode, setSettingsOpen, setUsageOpen, setCloudModelsOpen, theme.error])

  const composer = useComposerState(commandContext)
  sendRef.current = (text: string) => {
    if (!allowProviderMessageSend(modelsConfigured, showErrorToast)) return
    composer.handleSend(text)
  }

  const { interrupt, interruptAll } = useInterruptActions()
  const { rootRoleLabel, rootProfile } = useSlotProfiles()

  const rootTimeline = useDisplayState((state) => getFork(state, null) ?? null)
  const rootActor = useDisplayState((state) => state.actors["root"] ?? null)
  const displayMessages = useMemo(
    () => (rootTimeline ? orderedMessages(rootTimeline.messages) : []),
    [rootTimeline?.messages],
  )
  const context = rootActor?.context ?? null
  const contextHardCap = rootProfile?.contextWindow ?? null
  const tokenUsage = context && context.tokenEstimate > 0 ? context.tokenEstimate : null

  // Reserve footer width so attachments don't reflow when hints appear.
  const maxEscHintWidth = 'Press Esc again to interrupt all workers'.length
  const contextTextWidth = 24
  const attachmentsMaxWidth = Math.max(0, chatColumnWidth - 4 - maxEscHintWidth - contextTextWidth - 5)

  const composerCanFocus = !showRecentChats && !settingsOpen && !usageOpen && expandedForkStack.length === 0

  const submitUserMessage = useCallback((payload: {
    message: string
    visibleMessage?: string
    imageAttachments: RawImageAttachment[]
    mentions: RawMentionOccurrence[]
  }): void => {
    composer.handleSend(
      payload.message,
      { imageAttachments: payload.imageAttachments, mentions: payload.mentions },
      { visibleMessage: payload.visibleMessage },
    )
  }, [composer.handleSend])

  return (
    <Composer
      sessionId={sessionId}
      cwd={selectedCwd}
      status={rootTimeline?.mode ?? 'idle'}
      hasRunningForks={(rootActor?.work.activeChildCount ?? 0) > 0}
      bashMode={composer.bashMode}
      modelsConfigured={modelsConfigured}
      modelSummary={{
        role: rootRoleLabel,
        model: composer.model || '-',
        thinkingLevel: rootProfile?.reasoningEffort
          ? rootProfile.reasoningEffort.charAt(0).toUpperCase() + rootProfile.reasoningEffort.slice(1)
          : '-',
      }}
      tokenUsage={tokenUsage}
      contextHardCap={contextHardCap}
      isCompacting={context?.isCompacting ?? false}
      displayMode={displayMode}
      theme={theme}
      modeColor={theme.modeDefault}
      attachmentsMaxWidth={attachmentsMaxWidth}
      composerCanFocus={composerCanFocus}
      widgetNavActive={widgetNavActive}
      isWorkerView={false}
      enableAutopilot={false}
      autopilotEnabled={false}
      autopilotGenerating={false}
      submitUserMessage={submitUserMessage}
      runSlashCommand={composer.handleSlashCommand}
      executeBash={composer.handleRunBash}
      clearSystemBanners={clearSystemMessages}
      interruptFork={interrupt}
      interruptAll={interruptAll}
      openSettings={() => setSettingsOpen(true)}
      handleWidgetKeyEvent={handleWidgetKeyEvent}
      enterBashMode={() => setBashMode(true)}
      exitBashMode={() => setBashMode(false)}
      showToast={(message: string) => addEphemeralMessage(message, theme.error)}
      toggleAutopilot={() => { /* disabled */ }}
      displayMessages={displayMessages}
      selectedForkId={null}
      isBlockingOverlayActive={!composerCanFocus}
      selectedFileOpen={selectedFilePath !== null}
      onCloseFilePanel={() => setSelectedFilePath(null)}
    />
  )
}
