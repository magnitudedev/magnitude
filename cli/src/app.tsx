/**
 * CliApp — the orchestrator (spec §5.6, category: Orchestrator).
 *
 * Wires infrastructure (stream subscription, startup flow, terminal
 * keyboard, selection auto-copy), gates rendering (auth →
 * connection error → loading), and composes the feature containers into the
 * terminal layout. No feature logic, no rendering primitives beyond layout
 * boxes and the startup header slot.
 */
import { useCallback, type ReactNode } from "react";
import { Option } from "effect";
import {
  useAtomValue,
  useAtomSet,
  useAtomInitialValues,
  Result,
} from "@effect-atom/atom-react";
import {
  useSlotProfiles,
  useDisplayViewController,
  useDisplayConnectionError,
  useSelectedSessionId,
  usageOpenAtom,
  selectedFilePathAtom,
  selectedCwdAtom,
  sessionCreateOptionsAtom,
  useSessionPreload,
  useOnboardingModelSetup,
  isModelSlotConfigured,
  deriveLocalModelLoadActivity,
  notificationAreaStateAtom,
  deriveLocalModelPersistentNotificationStates,
  deriveSelectedModelResidencyNotificationState,
  notificationStatesEquivalent,
  resolveActiveNotificationState,
  deriveAcnRecoveryNotificationState,
  useServiceRecoveryState,
  useLocalModelsSelector,
  useModelSlots,
  useModelSlotActions,
} from "@magnitudedev/client-common";
import {
  type LocalModelsState,
  type SessionOptions,
} from "@magnitudedev/sdk";
import {
  authSourceAtom,
  modelMenuStateAtom,
  selectedFileSectionAtom,
  type AuthSource,
} from "./state/cli-atoms";
import {
  useSessionStartup,
  type SessionStart,
} from "./hooks/use-session-startup";
import { useTerminalKeyboard } from "./hooks/use-terminal-keyboard";
import { useTheme } from "./hooks/use-theme";
import { useLocalWidth } from "./hooks/use-local-width";
import { useSelectionAutoCopy } from "./utils/clipboard";
import { SelectedFileProvider } from "./hooks/use-file-viewer";
import { BOX_CHARS } from "./utils/ui-constants";
import type { ActionId } from "./types/ui-actions";

import { FatalErrorScreen } from "./features/app-shell/connection-error";
import { StartupHeader } from "./features/chat-timeline/startup-header";
import { Button } from "./components/button";
import { ChatTimelineContainer } from "./features/chat-timeline/container";
import { ComposerContainer } from "./features/composer/container";
import {
  ActivityRailContainer,
  TaskListContainer,
} from "./features/agent-status/container";
import {
  AppOverlaysContainer,
  useActiveOverlay,
} from "./features/overlays/container";
import { FileViewerPanelContainer } from "./features/file-viewer/container";
import { ModelMenusContainer } from "./features/model-menus/container";
import {
  useRecentChatsWidgetState,
  RecentChatsWidgetView,
} from "./features/sessions/container";
import { useSetupPresentation } from "./features/model-setup/setup-screen";
import { registerCliCommands } from "./commands/register";

registerCliCommands();

export type { SessionStart };

export interface CliAppProps {
  sessionStart: SessionStart;
  initialPrompt: string | undefined;
  envAuth: AuthSource;
  sessionOptions: SessionOptions;
}

export function CliApp(props: CliAppProps): ReactNode {
  useAtomInitialValues([
    [authSourceAtom, props.envAuth],
    [selectedCwdAtom, process.cwd()],
    [sessionCreateOptionsAtom, Option.some(props.sessionOptions)],
  ]);
  return <CliAppGates {...props} />;
}

function CliAppGates(props: CliAppProps): ReactNode {
  return (
    <CliEnvironmentGate>
      {(exitApp) => (
        <OnboardingGate
          {...props}
          onExitApp={exitApp}
        />
      )}
    </CliEnvironmentGate>
  );
}

function CliEnvironmentGate({
  children,
}: {
  readonly children: (exitApp: () => void) => ReactNode;
}): ReactNode {
  const connectionError = useDisplayConnectionError();
  const controller = useDisplayViewController();
  const exitApp = useCallback(() => {
    process.kill(process.pid, "SIGINT");
  }, []);

  if (connectionError && !connectionError.reconnecting) {
    return (
      <FatalErrorScreen
        error={connectionError.message}
        invariantViolation={connectionError.invariantViolation}
        onRetry={() => {
          const retried = controller.retry();
          if (!retried) {
            controller.clearSession();
          }
        }}
        onQuit={exitApp}
      />
    );
  }

  return children(exitApp);
}

function OnboardingGate(
  props: CliAppProps & {
    readonly onExitApp: () => void;
  }
): ReactNode {
  const onboardingSetup = useOnboardingModelSetup();
  const { slots, retry: retryProfiles } = useSlotProfiles();

  if (Result.isInitial(onboardingSetup.view)) return null;

  if (Result.isFailure(onboardingSetup.view)) {
    return (
      <FatalErrorScreen
        error="Failed to determine whether onboarding setup is required."
        onRetry={onboardingSetup.retry}
        onQuit={props.onExitApp}
      />
    );
  }

  const slotsSnapshot = Result.value(slots);
  if (Option.isNone(slotsSnapshot)) {
    if (Result.isFailure(slots)) {
      return (
        <FatalErrorScreen
          error="Failed to load model configuration from the service."
          onRetry={retryProfiles}
          onQuit={props.onExitApp}
        />
      );
    }
  }

  const primary = Option.map(slotsSnapshot, (state) => state.slots.primary);
  const onboardingSetupOpen = onboardingSetup.view.value._tag === "Open";
  const modelsConfigured = Option.exists(primary, isModelSlotConfigured);
  const modelsAvailableForInitialWork = Option.exists(primary, (slot) => {
    if (slot._tag === "Unassigned"
      || slot._tag === "Resolving"
      || slot.availability._tag !== "Available")
      return false;
    return true;
  });

  return (
    <CliAppContent
      {...props}
      modelsConfigured={modelsConfigured}
      modelsAvailableForInitialWork={modelsAvailableForInitialWork}
      onboardingSetupOpen={onboardingSetupOpen}
      onboardingSetup={onboardingSetup}
    />
  );
}

function CliAppContent(
  props: CliAppProps & {
    readonly modelsConfigured: boolean;
    readonly modelsAvailableForInitialWork: boolean;
    readonly onboardingSetupOpen: boolean;
    readonly onboardingSetup: ReturnType<typeof useOnboardingModelSetup>;
  }
): ReactNode {
  useSessionPreload(!props.onboardingSetupOpen);
  useSessionStartup({
    sessionStart: props.sessionStart,
    initialPrompt: props.initialPrompt,
    modelsConfigured:
      props.modelsAvailableForInitialWork && !props.onboardingSetupOpen,
  });

  const theme = useTheme();
  const sessionId = useSelectedSessionId();
  const selectedCwd = useAtomValue(selectedCwdAtom);
  const menu = useAtomValue(modelMenuStateAtom);
  const setMenu = useAtomSet(modelMenuStateAtom);
  const setUsageOpen = useAtomSet(usageOpenAtom);
  const activeOverlay = useActiveOverlay();
  const isOverlayActive = activeOverlay !== "none";

  const selectedFilePath = useAtomValue(selectedFilePathAtom);
  const selectedFileSection = useAtomValue(selectedFileSectionAtom);
  const selectedFile = selectedFilePath
    ? { path: selectedFilePath, section: selectedFileSection }
    : null;

  const widget = useRecentChatsWidgetState();
  const { showCopiedToast: clipboardToast } = useSelectionAutoCopy();
  const notificationAreaState = useAtomValue(notificationAreaStateAtom);
  const acnRecoveryState = useServiceRecoveryState();
  const onboardingSetup = props.onboardingSetup;
  const modelSlotsState = Option.getOrNull(Result.value(useModelSlots()));
  const { rootSlotId } = useSlotProfiles();
  const selectedLocalProviderModelId = modelSlotsState?.slots.primary._tag
    === "ConfiguredLocal"
    ? modelSlotsState.slots.primary.selection.providerModelId
    : null;
  const selectPersistentNotificationStates = useCallback(
    (modelsState: LocalModelsState) =>
      deriveLocalModelPersistentNotificationStates(
        modelsState,
        selectedLocalProviderModelId,
      ),
    [selectedLocalProviderModelId],
  );
  const persistentNotificationStates = useLocalModelsSelector(
    selectPersistentNotificationStates,
    notificationStatesEquivalent,
  );
  const notificationState = resolveActiveNotificationState(
    notificationAreaState,
    [
      deriveAcnRecoveryNotificationState(acnRecoveryState),
      ...Option.getOrElse(persistentNotificationStates, () => []),
      deriveSelectedModelResidencyNotificationState(modelSlotsState, rootSlotId),
    ],
  );
  const localModelLoadActivity = modelSlotsState === null
    ? null
    : deriveLocalModelLoadActivity(modelSlotsState, rootSlotId);
  const slotActions = useModelSlotActions();
  const chatColumn = useLocalWidth();
  const chatColumnWidth = chatColumn.width ?? 80;
  const clientWorkingDirectory = process.cwd();
  const dispatchErrorAction = useCallback(
    (actionId: ActionId) => {
      switch (actionId) {
        case "open-settings":
          setMenu({ open: true, root: "models" });
          return;
        case "open-usage":
          setUsageOpen(true);
          return;
      }
    },
    [setMenu, setUsageOpen]
  );

  useTerminalKeyboard({
    dispatchErrorAction,
    recentChatsEnabled: !props.onboardingSetupOpen,
    setupActive: props.onboardingSetupOpen,
  });

  const setupPresentation = useSetupPresentation(onboardingSetup, chatColumnWidth, props.onboardingSetupOpen);
  const setupSurface = setupPresentation.surface;
  const modelSetupPlaceholder = setupPresentation.placeholder;
  if (props.onboardingSetupOpen) {
    return (
      <box ref={chatColumn.ref} onSizeChange={chatColumn.onSizeChange} style={{ width: "100%", height: "100%", flexDirection: "column", padding: 1 }}>
        {setupSurface}
      </box>
    );
  }
  const activityRail = (
    <ActivityRailContainer
      modelLoadActivity={localModelLoadActivity}
      onStopModel={slotActions.stop}
      width={chatColumnWidth}
      agentActivityEnabled={!props.onboardingSetupOpen}
    />
  );

  // Startup header content — rendered inside the timeline scrollback.
  const startupHeader = (
    <StartupHeader
      width={chatColumnWidth}
      workingDirectory={clientWorkingDirectory.replace(
        process.env.HOME || "",
        "~"
      )}
      recentChats={
        !props.onboardingSetupOpen &&
        !widget.hasActivity &&
        !(menu.open && sessionId === null) ? (
          <RecentChatsWidgetView state={widget} />
        ) : null
      }
    />
  );

  return (
    <SelectedFileProvider value={selectedFile}>
      {isOverlayActive && (
        <AppOverlaysContainer dispatchErrorAction={dispatchErrorAction} />
      )}
      <box
        style={{
          visible: !isOverlayActive,
          flexDirection: "row",
          height: "100%",
        }}
      >
        <box
          ref={chatColumn.ref}
          onSizeChange={chatColumn.onSizeChange}
          style={{
            flexDirection: "column",
            flexGrow: 1,
            minWidth: 0,
            position: "relative",
            height: "100%",
          }}
        >
          <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column" }}>
            <box style={{ flexGrow: 1, minHeight: 0, flexDirection: "column" }}>
              <box
                style={{
                  flexGrow: 1,
                  flexShrink: 1,
                  minHeight: 0,
                  flexDirection: "column",
                  overflow: "hidden",
                }}
              >
                <ChatTimelineContainer
                  header={startupHeader}
                  chatColumnWidth={chatColumnWidth}
                  dispatchErrorAction={dispatchErrorAction}
                  isOverlayActive={isOverlayActive}
                  emptyState={setupSurface}
                  exclusiveEmptyState={props.onboardingSetupOpen}
                />
                {!props.onboardingSetupOpen && (
                  <box
                    style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}
                  >
                    <TaskListContainer />
                  </box>
                )}
                {props.onboardingSetupOpen ? (
                  <box
                    style={{
                      height: 1,
                      minHeight: 1,
                      maxHeight: 1,
                      flexShrink: 0,
                    }}
                  >
                    {activityRail}
                  </box>
                ) : (
                  activityRail
                )}
              </box>
              {menu.open && !props.onboardingSetupOpen ? (
                <ModelMenusContainer notificationState={notificationState} />
              ) : (
                <ComposerContainer
                  chatColumnWidth={chatColumnWidth}
                  clientWorkingDirectory={clientWorkingDirectory}
                  widgetNavActive={
                    widget.widgetNavActive && !props.onboardingSetupOpen
                  }
                  handleWidgetKeyEvent={widget.navigation.handleKeyEvent}
                  modelsConfigured={props.modelsConfigured}
                  modelSetupInProgress={props.onboardingSetupOpen}
                  modelSetupPlaceholder={
                    props.onboardingSetupOpen ? modelSetupPlaceholder : null
                  }
                  notificationState={notificationState}
                  openSetup={onboardingSetup.open}
                />
              )}
            </box>
          </box>

          {clipboardToast && (
            <Toast
              color={theme.status.success}
              background={theme.background.surface}
              text="Copied to clipboard"
            />
          )}
        </box>

        <FileViewerPanelContainer cwd={selectedCwd} />
      </box>
    </SelectedFileProvider>
  );
}

/** Bottom-right toast — pure layout primitive for the app shell. */
function Toast({
  color,
  background,
  text,
}: {
  color: string;
  background: string;
  text: string;
}): ReactNode {
  return (
    <box style={{ position: "absolute", bottom: 1, right: 2 }}>
      <box
        style={{
          borderStyle: "single",
          border: ["left"],
          borderColor: color,
          customBorderChars: { ...BOX_CHARS, vertical: "┃" },
        }}
      >
        <box
          style={{
            backgroundColor: background,
            paddingTop: 1,
            paddingBottom: 1,
            paddingLeft: 2,
            paddingRight: 2,
          }}
        >
          <text style={{ fg: color }}>{text}</text>
        </box>
      </box>
    </box>
  );
}
