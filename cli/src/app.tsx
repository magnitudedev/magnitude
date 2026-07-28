/**
 * CliApp — the orchestrator (spec §5.6, category: Orchestrator).
 *
 * Wires infrastructure (stream subscription, startup flow, terminal
 * keyboard, selection auto-copy), gates rendering (windows → auth →
 * connection error → loading), and composes the feature containers into the
 * terminal layout. No feature logic, no rendering primitives beyond layout
 * boxes and the startup header slot.
 */
import { useCallback, useSyncExternalStore, type ReactNode } from 'react'
import { Cause, Option } from 'effect'
import { useAtomValue, useAtomSet, useAtomInitialValues, Result } from '@effect-atom/atom-react'
import {
  useOnboardingState,
  useSlotProfiles,
  useDisplayViewController,
  useDisplayConnectionError,
  useSelectedSessionId,
  usageOpenAtom,
  selectedFilePathAtom,
  selectedCwdAtom,
  sessionCreateOptionsAtom,
  useSessionPreload,
  subscribeEphemeralMessage,
  getEphemeralMessageSnapshot,
  useFileWatchBridge,
  useOnboardingModelSetup,
  isModelSlotConfigured,
  deriveLocalModelLoadActivity,
} from '@magnitudedev/client-common'
import { PRIMARY_SLOT_ID, ReasoningEffortSchema, type SessionOptions } from '@magnitudedev/sdk'
import { authSourceAtom, modelMenuStateAtom, selectedFileSectionAtom, type AuthSource } from './state/cli-atoms'
import { useSessionStartup, type SessionStart } from './hooks/use-session-startup'
import { useTerminalBgDetection } from './hooks/use-terminal-bg-detection'
import { useTerminalKeyboard } from './hooks/use-terminal-keyboard'
import { useTheme } from './hooks/use-theme'
import { useLocalWidth } from './hooks/use-local-width'
import { useSelectionAutoCopy } from './utils/clipboard'
import { SelectedFileProvider } from './hooks/use-file-viewer'
import { BOX_CHARS } from './utils/ui-constants'
import type { ActionId } from './types/ui-actions'

import { FatalErrorScreen } from './features/app-shell/connection-error'
import { WindowsWarningScreen } from './features/app-shell/windows-warning'
import { StartupHeader } from './features/chat-timeline/startup-header'
import { Button } from './components/button'
import { ChatTimelineContainer } from './features/chat-timeline/container'
import { ComposerContainer } from './features/composer/container'
import { ActivityRailContainer, TaskListContainer } from './features/agent-status/container'
import { AppOverlaysContainer, useActiveOverlay } from './features/overlays/container'
import { FileViewerPanelContainer } from './features/file-viewer/container'
import { ModelMenusContainer } from './features/model-menus/container'
import { useRecentChatsWidgetState, RecentChatsWidgetView } from './features/sessions/container'
import {
  OnboardingModelChooser,
  OnboardingModelPreparation,
  deriveModelSetupActive,
  deriveOnboardingModelSetupView,
  onboardingModelSetupPlaceholder,
} from './features/model-setup'
import { registerCliCommands } from './commands/register'

registerCliCommands()

export type { SessionStart }

export interface CliAppProps {
  sessionStart: SessionStart
  initialPrompt: string | undefined
  goal: string | undefined
  envAuth: AuthSource
  sessionOptions: SessionOptions
  forceLocalInferenceSetup?: boolean
}

export function CliApp(props: CliAppProps): ReactNode {
  useAtomInitialValues([
    [authSourceAtom, props.envAuth],
    [selectedCwdAtom, process.cwd()],
    [sessionCreateOptionsAtom, Option.some(props.sessionOptions)],
  ])
  return <CliAppGates {...props} />
}

function CliAppGates(props: CliAppProps): ReactNode {
  return (
    <CliEnvironmentGate>
      {(exitApp) => (
        <OnboardingGate
          {...props}
          onExitApp={exitApp}
          forceSetup={props.forceLocalInferenceSetup ?? false}
        />
      )}
    </CliEnvironmentGate>
  )
}

function CliEnvironmentGate({ children }: {
  readonly children: (exitApp: () => void) => ReactNode
}): ReactNode {
  const connectionError = useDisplayConnectionError()
  const controller = useDisplayViewController()
  useTerminalBgDetection()

  const exitApp = useCallback(() => {
    process.kill(process.pid, 'SIGINT')
  }, [])

  if (process.platform === 'win32') {
    return <WindowsWarningScreen onExit={exitApp} />
  }

  if (connectionError && !connectionError.reconnecting) {
    return (
      <FatalErrorScreen
        error={connectionError.message}
        invariantViolation={connectionError.invariantViolation}
        onRetry={() => {
          const retried = controller.retry()
          if (!retried) {
            controller.clearSession()
          }
        }}
        onQuit={exitApp}
      />
    )
  }

  return children(exitApp)
}

function OnboardingGate(
  props: CliAppProps & {
    readonly onExitApp: () => void
    readonly forceSetup: boolean
  },
): ReactNode {
  const onboarding = useOnboardingState()
  const { slots, retry: retryProfiles } = useSlotProfiles()
  const controller = useDisplayViewController()

  if (Result.isFailure(onboarding.state)) {
    return (
      <FatalErrorScreen
        error="Failed to read onboarding state."
        onRetry={() => controller.retry()}
        onQuit={props.onExitApp}
      />
    )
  }

  const slotsSnapshot = Result.value(slots)
  if (Option.isNone(slotsSnapshot)) {
    if (Result.isFailure(slots)) {
      return (
        <FatalErrorScreen
          error="Failed to load model configuration from the daemon."
          onRetry={retryProfiles}
          onQuit={props.onExitApp}
        />
      )
    }
  }

  const onboardingRequired = Result.isSuccess(onboarding.state)
    ? !onboarding.state.value.completed
    : true
  const primary = Option.map(slotsSnapshot, ({ state }) => state.slots.primary)
  const modelSetupActive = deriveModelSetupActive({
    forceSetup: props.forceSetup,
    onboardingRequired,
    completionSucceeded: Result.isSuccess(onboarding.updateResult),
  })
  const modelsConfigured = Option.exists(primary, isModelSlotConfigured)
  const modelsReadyForInitialWork = Option.exists(primary, (slot) => {
    if (slot._tag === 'Unassigned' || slot.availability._tag !== 'Available') return false
    if (slot._tag === 'ConfiguredRemote') return true
    return Option.exists(slot.instance, (instance) => instance.lifecycle._tag === 'Ready')
  })

  return (
    <CliAppContent
      {...props}
      modelsConfigured={modelsConfigured}
      modelsReadyForInitialWork={modelsReadyForInitialWork}
      modelSetupActive={modelSetupActive}
      updateOnboarding={onboarding.update}
      updateOnboardingResult={onboarding.updateResult}
    />
  )
}

function CliAppContent(props: CliAppProps & {
  readonly modelsConfigured: boolean
  readonly modelsReadyForInitialWork: boolean
  readonly modelSetupActive: boolean
  readonly updateOnboarding: ReturnType<typeof useOnboardingState>["update"]
  readonly updateOnboardingResult: ReturnType<typeof useOnboardingState>["updateResult"]
}): ReactNode {
  useSessionPreload(!props.modelSetupActive)
  useFileWatchBridge()
  useSessionStartup({
    sessionStart: props.sessionStart,
    initialPrompt: props.initialPrompt,
    goal: props.goal,
    modelsConfigured: props.modelsReadyForInitialWork && !props.modelSetupActive,
  })

  const theme = useTheme()
  const sessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const menu = useAtomValue(modelMenuStateAtom)
  const setMenu = useAtomSet(modelMenuStateAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const activeOverlay = useActiveOverlay()
  const isOverlayActive = activeOverlay !== 'none'

  const selectedFilePath = useAtomValue(selectedFilePathAtom)
  const selectedFileSection = useAtomValue(selectedFileSectionAtom)
  const selectedFile = selectedFilePath ? { path: selectedFilePath, section: selectedFileSection } : null

  const widget = useRecentChatsWidgetState()
  const { showCopiedToast: clipboardToast } = useSelectionAutoCopy()
  const ephemeralMessage = useSyncExternalStore(subscribeEphemeralMessage, getEphemeralMessageSnapshot)
  const onboardingSetup = useOnboardingModelSetup(props.updateOnboarding)
  const local = onboardingSetup.local
  const localInference = local.state
  const localInferenceSnapshot = Result.value(localInference)
  const localInferenceView = onboardingSetup.view
  const downloadingModelCount = Option.match(localInferenceSnapshot, {
    onNone: () => 0,
    onSome: ({ models }) => models.models.filter((model) => model.download._tag === 'Downloading').length,
  })
  const downloadSummary = downloadingModelCount === 0
    ? null
    : `${downloadingModelCount} ${downloadingModelCount === 1 ? 'model' : 'models'} downloading`
  const { rootSlotId } = useSlotProfiles()
  const localModelLoadActivity = Option.match(localInferenceSnapshot, {
    onNone: () => null,
    onSome: (state) => deriveLocalModelLoadActivity(state.slots, rootSlotId),
  })
  const cancelDownloadError = Result.isFailure(local.cancelDownloadResult)
    ? Cause.pretty(local.cancelDownloadResult.cause)
    : null
  const activationMutationError = Result.isFailure(local.loadModelResult)
    ? Cause.pretty(local.loadModelResult.cause)
    : Result.isFailure(local.slotAssignment)
      ? Cause.pretty(local.slotAssignment.cause)
      : null
  const completionMutationError = Result.isFailure(props.updateOnboardingResult)
    ? Cause.pretty(props.updateOnboardingResult.cause)
    : null
  const completeModelSetup = useCallback(() => {
    void props.updateOnboarding(true)
  }, [props.updateOnboarding])
  const selectOnboardingModel = onboardingSetup.select
  const cancelOnboardingModelSetup = onboardingSetup.cancel
  const chatColumn = useLocalWidth()
  const chatColumnWidth = chatColumn.width ?? 80
  const clientWorkingDirectory = process.cwd()
  const dispatchErrorAction = useCallback((actionId: ActionId) => {
    switch (actionId) {
      case 'open-settings':
        setMenu({ open: true, root: 'models' })
        return
      case 'open-usage':
        setUsageOpen(true)
        return
    }
  }, [setMenu, setUsageOpen])

  useTerminalKeyboard({
    dispatchErrorAction,
    recentChatsEnabled: !props.modelSetupActive,
  })

  const setupView = deriveOnboardingModelSetupView({
    active: props.modelSetupActive,
    submittedProviderModelId: onboardingSetup.submittedProviderModelId,
    state: localInferenceView,
  })
  const setupSurface = (() => {
    switch (setupView._tag) {
      case 'Inactive': return undefined
      case 'Preparing': return (
        <OnboardingModelPreparation
          state={setupView.state}
          width={chatColumnWidth}
          onSkip={completeModelSetup}
        />
      )
      case 'Downloading': return (
        <OnboardingModelChooser
          state={setupView.state}
          width={chatColumnWidth}
          pending={false}
          error={null}
          operation={{
            _tag: 'Downloading',
            candidate: setupView.candidate,
            cancelling: Result.isWaiting(local.cancelDownloadResult),
            cancelError: cancelDownloadError,
            onCancel: cancelOnboardingModelSetup,
            onRetry: () => selectOnboardingModel({
              targetId: setupView.candidate.targetId,
              providerModelId: setupView.candidate.providerModelId,
              reasoningEffort: Option.getOrElse(
                setupView.candidate.capabilities.reasoning.defaultEffort,
                () => ReasoningEffortSchema.make("none"),
              ),
            }),
          }}
          onChoose={selectOnboardingModel}
          onContinue={completeModelSetup}
          onSkip={completeModelSetup}
        />
      )
      case 'Activating': return (
        <OnboardingModelChooser
          state={setupView.state}
          width={chatColumnWidth}
          pending={false}
          error={null}
          operation={{
            _tag: 'Activating',
            providerModelId: setupView.providerModelId,
            displayName: setupView.displayName,
            phase: setupView.phase === "Ready" && completionMutationError !== null
              ? "Failed"
              : setupView.phase === "Preparing" && activationMutationError !== null
              ? "Failed"
              : setupView.phase,
            failure: setupView.failure
              ?? (setupView.phase === "Ready"
                ? completionMutationError
                : activationMutationError),
            onRetry: () => {
              const candidate = setupView.state.models.recommendations._tag === "Ready"
                ? setupView.state.models.recommendations.catalog.find(({ providerModelId }) =>
                    providerModelId === setupView.providerModelId)
                : undefined
              if (candidate) {
                void selectOnboardingModel({
                  targetId: candidate.targetId,
                  providerModelId: candidate.providerModelId,
                  reasoningEffort: Option.getOrElse(
                    candidate.capabilities.reasoning.defaultEffort,
                    () => ReasoningEffortSchema.make("none"),
                  ),
                })
              } else {
                void local.loadModel(PRIMARY_SLOT_ID)
              }
            },
            onChooseAnother: cancelOnboardingModelSetup,
          }}
          onChoose={selectOnboardingModel}
          onContinue={completeModelSetup}
          onSkip={completeModelSetup}
        />
      )
      case 'Choosing': return (
        <OnboardingModelChooser
          state={setupView.state}
          width={chatColumnWidth}
          pending={false}
          error={null}
          operation={null}
          onChoose={selectOnboardingModel}
          onContinue={completeModelSetup}
          onSkip={completeModelSetup}
        />
      )
    }
  })()
  const modelSetupPlaceholder = onboardingModelSetupPlaceholder(setupView)
  const activityRail = (
    <ActivityRailContainer
      modelLoadActivity={localModelLoadActivity}
      onStopModel={local.stopModel}
      width={chatColumnWidth}
      agentActivityEnabled={!props.modelSetupActive}
    />
  )

  // Startup header content — rendered inside the timeline scrollback.
  const startupHeader = (
    <StartupHeader
      width={chatColumnWidth}
      workingDirectory={clientWorkingDirectory.replace(process.env.HOME || '', '~')}
      recentChats={!props.modelSetupActive && !widget.hasActivity && !(menu.open && sessionId === null)
        ? <RecentChatsWidgetView state={widget} />
        : null}
    />
  )

  return (
    <SelectedFileProvider value={selectedFile}>
      {isOverlayActive && <AppOverlaysContainer dispatchErrorAction={dispatchErrorAction} />}
      <box style={{ visible: !isOverlayActive, flexDirection: 'row', height: '100%' }}>
        <box
          ref={chatColumn.ref}
          onSizeChange={chatColumn.onSizeChange}
          style={{ flexDirection: 'column', flexGrow: 1, minWidth: 0, position: 'relative', height: '100%' }}
        >
          <box style={{ flexGrow: 1, minHeight: 0, flexDirection: 'column' }}>
            <box style={{ flexGrow: 1, minHeight: 0, flexDirection: 'column' }}>
              <box style={{ flexGrow: 1, flexShrink: 1, minHeight: 0, flexDirection: 'column', overflow: 'hidden' }}>
                <ChatTimelineContainer
                  header={startupHeader}
                  chatColumnWidth={chatColumnWidth}
                  dispatchErrorAction={dispatchErrorAction}
                  isOverlayActive={isOverlayActive}
                  emptyState={setupSurface}
                  exclusiveEmptyState={props.modelSetupActive}
                />
                {!props.modelSetupActive && (
                  <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
                    <TaskListContainer />
                  </box>
                )}
                {props.modelSetupActive ? (
                  <box style={{ height: 1, minHeight: 1, maxHeight: 1, flexShrink: 0 }}>
                    {activityRail}
                  </box>
                ) : activityRail}
              </box>
              {menu.open && !props.modelSetupActive
                ? <ModelMenusContainer downloadSummary={downloadSummary} />
                : (
                  <ComposerContainer
                    chatColumnWidth={chatColumnWidth}
                    clientWorkingDirectory={clientWorkingDirectory}
                    widgetNavActive={widget.widgetNavActive && !props.modelSetupActive}
                    handleWidgetKeyEvent={widget.navigation.handleKeyEvent}
                    modelsConfigured={props.modelsConfigured}
                    modelSetupInProgress={props.modelSetupActive}
                    modelSetupPlaceholder={props.modelSetupActive ? modelSetupPlaceholder : null}
                  />
                )}
            </box>
          </box>

          {clipboardToast && (
            <Toast color={theme.success} background={theme.surface} text="Copied to clipboard" />
          )}
          {ephemeralMessage && (
            <Toast
              color={ephemeralMessage.color ?? (ephemeralMessage.tone === 'warning' ? theme.warning : theme.error)}
              background={theme.surface}
              text={ephemeralMessage.text}
            />
          )}
        </box>

        <FileViewerPanelContainer cwd={selectedCwd} />
      </box>
    </SelectedFileProvider>
  )
}

/** Bottom-right toast — pure layout primitive for the app shell. */
function Toast({ color, background, text }: { color: string; background: string; text: string }): ReactNode {
  return (
    <box style={{ position: 'absolute', bottom: 1, right: 2 }}>
      <box style={{
        borderStyle: 'single',
        border: ['left'],
        borderColor: color,
        customBorderChars: { ...BOX_CHARS, vertical: '┃' },
      }}>
        <box style={{
          backgroundColor: background,
          paddingTop: 1,
          paddingBottom: 1,
          paddingLeft: 2,
          paddingRight: 2,
        }}>
          <text style={{ fg: color }}>{text}</text>
        </box>
      </box>
    </box>
  )
}
