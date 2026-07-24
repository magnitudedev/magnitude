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
import { Option } from 'effect'
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
  useLocalInferenceQuery,
  isModelSlotUsableForMessages,
} from '@magnitudedev/client-common'
import { PRIMARY_SLOT_ID, type SessionOptions } from '@magnitudedev/sdk'
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
import { AnimatedLogo } from './components/animated-logo'
import { Button } from './components/button'
import { ChatTimelineContainer } from './features/chat-timeline/container'
import { ComposerContainer } from './features/composer/container'
import { WorkingTimerContainer, TaskListContainer } from './features/agent-status/container'
import { AppOverlaysContainer, useActiveOverlay } from './features/overlays/container'
import { FileViewerPanelContainer } from './features/file-viewer/container'
import { LocalInferenceStatusBar } from './features/local-inference/status-bar'
import { ModelMenusContainer } from './features/model-menus/container'
import { useRecentChatsWidgetState, RecentChatsWidgetView } from './features/sessions/container'
import {
  ModelSetupScreen,
  PreparingModelSetupScreen,
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
    return (
      <ModelSetupScreen
        mode="onboarding"
        onExit={props.onExitApp}
        onComplete={() => onboarding.complete("model_setup")}
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

  const primary = slotsSnapshot.value.state.slots.primary
  const modelsConfigured = primary.slotId === PRIMARY_SLOT_ID
    && isModelSlotUsableForMessages(primary)

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
  const localInference = useLocalInferenceQuery()
  const localInferenceSnapshot = Result.value(localInference)
  const downloadingModelCount = Option.match(localInferenceSnapshot, {
    onNone: () => 0,
    onSome: ({ models }) => models.models.filter((model) => model.download._tag === 'Downloading').length,
  })
  const downloadSummary = downloadingModelCount === 0
    ? null
    : `${downloadingModelCount} ${downloadingModelCount === 1 ? 'model' : 'models'} downloading`
  const { rootProfile } = useSlotProfiles()
  const chatColumn = useLocalWidth()
  const chatColumnWidth = chatColumn.width ?? 80

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
        <text style={{ fg: theme.muted }}> to choose and manage models.</text>
      </box>
      {!widget.hasActivity && !(menu.open && sessionId === null) && (
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
            <LocalInferenceStatusBar
              state={Option.getOrNull(localInferenceSnapshot)}
              width={chatColumnWidth}
              selectedModelName={rootProfile?.modelDisplayName ?? null}
              selectedProviderId={rootProfile?.providerId ?? null}
              onOpenModels={() => setMenu({ open: true, root: 'models' })}
              onOpenHardware={() => setMenu({ open: true, root: 'hardware' })}
            />
            <box style={{ flexGrow: 1, minHeight: 0, flexDirection: 'column' }}>
              <box style={{ flexGrow: 1, flexShrink: 1, minHeight: 0, flexDirection: 'column', overflow: 'hidden' }}>
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
              </box>
              {menu.open
                ? <ModelMenusContainer downloadSummary={downloadSummary} />
                : (
                  <ComposerContainer
                    chatColumnWidth={chatColumnWidth}
                    widgetNavActive={widget.widgetNavActive}
                    handleWidgetKeyEvent={widget.navigation.handleKeyEvent}
                    modelsConfigured={props.modelsConfigured}
                    downloadSummary={downloadSummary}
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
