/**
 * CliApp — the orchestrator (spec §5.6, category: Orchestrator).
 *
 * Wires infrastructure (stream subscription, startup flow, terminal
 * keyboard, selection auto-copy), gates rendering (windows → auth →
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
  useLocalModelsSelector,
  useModelSlots,
  useModelSlotActions,
  useAcnLifecycle,
  formatLocalModelDisplayName,
  onboardingModelSetupFailureMessage,
  type OnboardingModelSetupState,
} from "@magnitudedev/client-common";
import {
  type LocalModelsState,
  type SessionOptions,
  type AcnLifecycleState,
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
import { WindowsWarningScreen } from "./features/app-shell/windows-warning";
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
import {
  OnboardingModelChooser,
  OnboardingModelExiting,
  OnboardingModelPreparation,
} from "./features/model-setup";
import { registerCliCommands } from "./commands/register";
import { AcnBootstrapScreen } from "./features/app-shell/acn-bootstrap";

registerCliCommands();

export type { SessionStart };

export interface CliAppProps {
  sessionStart: SessionStart;
  initialPrompt: string | undefined;
  goal: string | undefined;
  envAuth: AuthSource;
  sessionOptions: SessionOptions;
  initialAcnLifecycle: AcnLifecycleState;
}

export function CliApp(props: CliAppProps): ReactNode {
  useAtomInitialValues([
    [authSourceAtom, props.envAuth],
    [selectedCwdAtom, process.cwd()],
    [sessionCreateOptionsAtom, Option.some(props.sessionOptions)],
  ]);
  const { state, retry } = useAcnLifecycle(props.initialAcnLifecycle);
  return state._tag === "Ready" ? (
    <CliAppGates {...props} />
  ) : (
    <AcnBootstrapScreen
      state={state}
      onRetry={retry}
      onQuit={() => process.kill(process.pid, "SIGINT")}
    />
  );
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

  if (process.platform === "win32") {
    return <WindowsWarningScreen onExit={exitApp} />;
  }

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
          error="Failed to load model configuration from the daemon."
          onRetry={retryProfiles}
          onQuit={props.onExitApp}
        />
      );
    }
  }

  const primary = Option.map(slotsSnapshot, ({ state }) => state.slots.primary);
  const onboardingSetupOpen = onboardingSetup.view.value._tag === "Open";
  const modelsConfigured = Option.exists(primary, isModelSlotConfigured);
  const modelsAvailableForInitialWork = Option.exists(primary, (slot) => {
    if (slot._tag === "Unassigned" || slot.availability._tag !== "Available")
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
    goal: props.goal,
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
  const onboardingSetup = props.onboardingSetup;
  const setupState = Option.getOrNull(Result.value(onboardingSetup.view));
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
      ...Option.getOrElse(persistentNotificationStates, () => []),
      deriveSelectedModelResidencyNotificationState(modelSlotsState, rootSlotId),
    ],
  );
  const localModelLoadActivity = modelSlotsState === null
    ? null
    : deriveLocalModelLoadActivity(modelSlotsState, rootSlotId);
  const setupOnboardingModel = onboardingSetup.select;
  const cancelOnboardingModelSetup = onboardingSetup.cancel;
  const exitOnboardingModelSetup = onboardingSetup.exit;
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
  });

  const setupPreparation = (
    progress: LocalModelsState["discoveryState"]["progress"],
    error: string | null,
    exitKind: "Skip" | "Close" | null,
  ) => ({
    surface: (
      <OnboardingModelPreparation
        hardware={onboardingSetup.hardware}
        progress={progress}
        error={error}
        width={chatColumnWidth}
        onExit={exitKind === null ? undefined : exitOnboardingModelSetup}
        exitKind={exitKind}
      />
    ),
    placeholder: "Preparing local models…",
  });
  const setupWithState = (state: OnboardingModelSetupState) => {
    if (state._tag === "Closed") {
      return { surface: undefined, placeholder: null };
    }
    if (state.content._tag === "Closing") {
      return {
        surface: (
          <OnboardingModelExiting
            hardware={onboardingSetup.hardware}
            width={chatColumnWidth}
          />
        ),
        placeholder: "Finishing onboarding…",
      };
    }
    if (state.content._tag === "Preparation") {
      const content = state.content;
      const notice = Option.map(
        state.notice,
        onboardingModelSetupFailureMessage,
      );
      return setupPreparation(
        content.progress,
        Option.getOrElse(
          notice,
          () => content.discoveryFailure?.message ?? null,
        ),
        state.exitKind,
      );
    }
    const content = state.content;
    const setupError = Option.match(state.notice, {
      onNone: () => null,
      onSome: onboardingModelSetupFailureMessage,
    });
    const chooser = (
      operation: Parameters<typeof OnboardingModelChooser>[0]["operation"],
      placeholder: string,
    ) => ({
      surface: (
        <OnboardingModelChooser
          hardware={onboardingSetup.hardware}
          options={content.options}
          width={chatColumnWidth}
          error={setupError}
          operation={operation}
          onSelect={setupOnboardingModel}
          onExit={exitOnboardingModelSetup}
          exitKind={state.exitKind}
        />
      ),
      placeholder,
    });
    return Option.match(content.operation, {
      onNone: () => chooser(null, "Select a model to start coding…"),
      onSome: (operation) => {
        switch (operation._tag) {
          case "Preparing": {
            const acquisition = operation.model.acquisitionState;
            if (acquisition._tag !== "Installed") {
              const starting = acquisition._tag !== "Downloading";
              return chooser({
                _tag: "Downloading",
                model: operation.model,
                starting,
                cancelling: operation.cancelling,
                onCancel: cancelOnboardingModelSetup,
              }, `${starting ? "Starting download for" : "Downloading"} ${formatLocalModelDisplayName(operation.model)}…`);
            }
            return chooser({
              _tag: "Configuring",
              model: operation.model,
            }, `Configuring ${formatLocalModelDisplayName(operation.model)}…`);
          }
          case "Configuring": return chooser({
            _tag: "Configuring",
            model: operation.model,
          }, `Configuring ${formatLocalModelDisplayName(operation.model)}…`);
          case "Installing": return chooser({
            _tag: "Downloading",
            model: operation.model,
            starting: operation.model.acquisitionState._tag === "NotInstalled",
            cancelling: operation.cancelling,
            onCancel: cancelOnboardingModelSetup,
          }, `Downloading ${formatLocalModelDisplayName(operation.model)}…`);
          case "Loading": return chooser({
            _tag: "Activating",
            providerModelId: operation.providerModelId,
            model: operation.model,
            phase: operation.phase,
            failure: operation.failure,
            onRetry: () => setupOnboardingModel(operation.modelId),
            onChooseAnother: cancelOnboardingModelSetup,
          }, operation.phase === "Loading"
            ? `Loading ${formatLocalModelDisplayName(operation.model)}…`
            : operation.phase === "Stopping"
              ? `Stopping ${formatLocalModelDisplayName(operation.model)}…`
              : operation.phase === "Ready"
                ? `Finishing setup for ${formatLocalModelDisplayName(operation.model)}…`
                : `Couldn’t load ${formatLocalModelDisplayName(operation.model)}`);
          case "Completing": return chooser({
            _tag: "Activating",
            providerModelId: operation.providerModelId,
            model: operation.model,
            phase: "Ready",
            failure: null,
            onRetry: () => setupOnboardingModel(operation.modelId),
            onChooseAnother: cancelOnboardingModelSetup,
          }, `Finishing setup for ${formatLocalModelDisplayName(operation.model)}…`);
        }
      },
    });
  };
  const setupPresentation = !props.onboardingSetupOpen
    ? { surface: undefined, placeholder: null }
    : Result.match(onboardingSetup.view, {
        onInitial: () => setupPreparation([], null, null),
        onFailure: () => setupPreparation([], "Local model setup is unavailable.", null),
        onSuccess: ({ value }) => setupWithState(value),
      });
  const setupSurface = setupPresentation.surface;
  const modelSetupPlaceholder = setupPresentation.placeholder;
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
