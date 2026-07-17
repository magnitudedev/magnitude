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
import { TextAttributes } from '@opentui/core'
import { Cause, Option } from 'effect'
import { useAtomValue, useAtomSet, useAtomInitialValues, Result } from '@effect-atom/atom-react'
import {
  useOnboardingState,
  useSlotProfiles,
  useDisplayViewController,
  useDisplayConnectionError,
  useSelectedSessionId,
  settingsOpenAtom,
  usageOpenAtom,
  selectedFilePathAtom,
  selectedCwdAtom,
  sessionCreateOptionsAtom,
  useSessionPreload,
  subscribeEphemeralMessage,
  getEphemeralMessageSnapshot,
  useFileWatchBridge,
  useLocalInferenceQuery,
} from '@magnitudedev/client-common'
import { ModelSlotsLifecycle, type SessionOptions } from '@magnitudedev/sdk'
import { authSourceAtom, selectedFileSectionAtom, type AuthSource } from './state/cli-atoms'
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
import { AnimatedLogo } from './components/animated-logo'
import { ChatTimelineContainer } from './features/chat-timeline/container'
import { ComposerContainer } from './features/composer/container'
import { WorkingTimerContainer, TaskListContainer } from './features/agent-status/container'
import { AppOverlaysContainer, useActiveOverlay } from './features/overlays/container'
import { FileViewerPanelContainer } from './features/file-viewer/container'
import { useRecentChatsWidgetState, RecentChatsWidgetView } from './features/sessions/container'
import {
  ModelSetupScreen,
  PreparingModelSetupScreen,
} from './features/model-setup'
import { blockingModelSlotsFailure } from './features/model-setup/model-slots-gate'

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
  const { profiles, slots, retry: retryProfiles } = useSlotProfiles()
  const controller = useDisplayViewController()

  if (Result.isInitial(onboarding.state)) {
    return <PreparingModelSetupScreen />
  }

  if (Result.isFailure(onboarding.state)) {
    return (
      <FatalErrorScreen
        error="Failed to read onboarding state."
        onRetry={() => controller.retry()}
        onQuit={props.onExitApp}
      />
    )
  }

  const onboardingRequired = onboarding.state.value.flows.model_setup.required
  const forcedSetupComplete = props.forceSetup && Result.isSuccess(onboarding.completeResult)
  if (onboardingRequired || (props.forceSetup && !forcedSetupComplete)) {
    const completionError = Result.isFailure(onboarding.completeResult)
      ? Cause.pretty(onboarding.completeResult.cause)
      : null
    return (
      <ModelSetupScreen
        mode="onboarding"
        onExit={props.onExitApp}
        onComplete={() => onboarding.complete("model_setup")}
        completing={Result.isWaiting(onboarding.completeResult)}
        completionError={completionError}
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
    return <PreparingModelSetupScreen />
  }

  const slotsState = slotsSnapshot.value.state
  if (ModelSlotsLifecycle.is(slotsState, 'loading')) {
    return <PreparingModelSetupScreen />
  }

  if (ModelSlotsLifecycle.is(slotsState, 'unavailable')) {
    const blockingFailure = blockingModelSlotsFailure(slotsState)
    if (blockingFailure) {
      return (
        <FatalErrorScreen
          error={`Failed to load model configuration: ${blockingFailure}`}
          onRetry={retryProfiles}
          onQuit={props.onExitApp}
        />
      )
    }
  }

  const modelsConfigured = Boolean(profiles?.primary)

  return (
    <CliAppContent
      {...props}
      modelsConfigured={modelsConfigured}
    />
  )
}

function CliAppContent(props: CliAppProps & { readonly modelsConfigured: boolean }): ReactNode {
  useSessionPreload()
  useFileWatchBridge()
  useSessionStartup({
    sessionStart: props.sessionStart,
    initialPrompt: props.initialPrompt,
    goal: props.goal,
    modelsConfigured: props.modelsConfigured,
  })

  const theme = useTheme()
  const sessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const activeOverlay = useActiveOverlay()
  const isOverlayActive = activeOverlay !== 'none'

  const selectedFilePath = useAtomValue(selectedFilePathAtom)
  const selectedFileSection = useAtomValue(selectedFileSectionAtom)
  const selectedFile = selectedFilePath ? { path: selectedFilePath, section: selectedFileSection } : null

  const widget = useRecentChatsWidgetState()
  const { showCopiedToast: clipboardToast } = useSelectionAutoCopy()
  const ephemeralMessage = useSyncExternalStore(subscribeEphemeralMessage, getEphemeralMessageSnapshot)
  const localInference = useLocalInferenceQuery()
  const localInferenceSnapshot = Result.value(localInference)
  const loadedLocalModels = Option.getOrElse(Option.map(localInferenceSnapshot, (snapshot) =>
    snapshot.choices.filter((choice) =>
        (choice._tag === 'RunningExternal' || choice._tag === 'RunningManaged')
        && (choice.residency === 'loaded' || choice.residency === 'sleeping'))), () => [])
  const loadingLocalModels = Option.getOrElse(Option.map(localInferenceSnapshot, (snapshot) =>
    snapshot.operations.filter((operation) => operation.status === 'running')), () => [])

  const chatColumn = useLocalWidth()
  const chatColumnWidth = chatColumn.width ?? 80

  const dispatchErrorAction = useCallback((actionId: ActionId) => {
    switch (actionId) {
      case 'open-settings':
        setSettingsOpen(true)
        return
      case 'open-usage':
        setUsageOpen(true)
        return
    }
  }, [setSettingsOpen, setUsageOpen])

  useTerminalKeyboard({ dispatchErrorAction })

  // Startup header content — rendered inside the timeline scrollback.
  const startupHeader = (
    <>
      <box style={{ paddingLeft: 1, paddingBottom: 1 }}>
        <AnimatedLogo />
      </box>
      <box style={{ paddingLeft: 1, flexDirection: 'row' }}>
        <text style={{ fg: theme.muted }}>Current directory: </text>
        <text style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>
          {process.cwd().replace(process.env.HOME || '', '~')}
        </text>
      </box>
      <box style={{ paddingLeft: 1, paddingBottom: widget.hasActivity ? 1 : 0, flexDirection: 'row' }}>
        <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Tip: </text>
        <text style={{ fg: theme.muted }}>Use </text>
        <text style={{ fg: theme.foreground }}>/settings</text>
        <text style={{ fg: theme.muted }}> to view your connection and roles.</text>
      </box>
      {!widget.hasActivity && (
        <box style={{ paddingLeft: 1 }}>
          <RecentChatsWidgetView state={widget} />
        </box>
      )}
    </>
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
            <ChatTimelineContainer
              header={startupHeader}
              chatColumnWidth={chatColumnWidth}
              dispatchErrorAction={dispatchErrorAction}
              isOverlayActive={isOverlayActive}
            />
            <WorkingTimerContainer />
            <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
              <TaskListContainer />
            </box>
            <ComposerContainer
              chatColumnWidth={chatColumnWidth}
              widgetNavActive={widget.widgetNavActive}
              handleWidgetKeyEvent={widget.navigation.handleKeyEvent}
              modelsConfigured={props.modelsConfigured}
            />
            {(loadedLocalModels.length > 0 || loadingLocalModels.length > 0) && (
              <box style={{ paddingLeft: 2, paddingRight: 2, flexShrink: 0 }}>
                <text style={{ fg: theme.muted }}>
                  {[
                    loadedLocalModels.length > 0 ? `Local loaded: ${loadedLocalModels.map((model) => model.displayName).join(' · ')}` : null,
                    ...loadingLocalModels.map((operation) => `Loading ${operation.providerModelId}: ${operation.stage}${operation.progress === undefined ? '' : ` ${Math.round(operation.progress * 100)}%`}`),
                  ].filter(Boolean).join('  ·  ')}
                </text>
              </box>
            )}
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
