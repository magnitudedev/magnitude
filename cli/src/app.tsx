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
import { Cause, Option, Schema } from "effect";
import {
  useAtomValue,
  useAtomSet,
  useAtomInitialValues,
  Result,
} from "@effect-atom/atom-react";
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
  useFileWatchBridge,
  useOnboardingModelSetup,
  isModelSlotConfigured,
  deriveLocalModelLoadActivity,
  notificationAreaStateAtom,
  deriveLocalModelPersistentNotificationStates,
  notificationStatesEquivalent,
  resolveActiveNotificationState,
  useLocalModelsSelector,
  useModelSlots,
  useModelSlotActions,
  useAcnLifecycle,
  localModelConfigurationId,
  formatLocalModelDisplayName,
  type OnboardingModelSetupState,
} from "@magnitudedev/client-common";
import {
  ModelDownloadFailureSchema,
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
import { useTerminalBgDetection } from "./hooks/use-terminal-bg-detection";
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
  OnboardingModelPreparation,
} from "./features/model-setup";
import { modelDownloadFailureMessage } from "./features/local-inference/view-model";
import { registerCliCommands } from "./commands/register";
import { AcnBootstrapScreen } from "./features/app-shell/acn-bootstrap";

registerCliCommands();

export type { SessionStart };

const modelSetupIsActive = ({
  forceSetup,
  onboardingRequired,
  completionSucceeded,
}: {
  readonly forceSetup: boolean;
  readonly onboardingRequired: boolean;
  readonly completionSucceeded: boolean;
}): boolean => onboardingRequired || (forceSetup && !completionSucceeded);

export interface CliAppProps {
  sessionStart: SessionStart;
  initialPrompt: string | undefined;
  goal: string | undefined;
  envAuth: AuthSource;
  sessionOptions: SessionOptions;
  forceLocalInferenceSetup?: boolean;
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
          forceSetup={props.forceLocalInferenceSetup ?? false}
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
  useTerminalBgDetection();

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
    readonly forceSetup: boolean;
  }
): ReactNode {
  const onboarding = useOnboardingState();
  const { slots, retry: retryProfiles } = useSlotProfiles();
  const controller = useDisplayViewController();

  if (Result.isFailure(onboarding.state)) {
    return (
      <FatalErrorScreen
        error="Failed to read onboarding state."
        onRetry={() => controller.retry()}
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

  const onboardingRequired = Result.isSuccess(onboarding.state)
    ? !onboarding.state.value.completed
    : false;
  const primary = Option.map(slotsSnapshot, ({ state }) => state.slots.primary);
  const modelSetupActive = modelSetupIsActive({
    forceSetup: props.forceSetup,
    onboardingRequired,
    completionSucceeded: Result.isSuccess(onboarding.updateResult),
  });
  const modelsConfigured = Option.exists(primary, isModelSlotConfigured);
  const modelsReadyForInitialWork = Option.exists(primary, (slot) => {
    if (slot._tag === "Unassigned" || slot.availability._tag !== "Available")
      return false;
    if (slot._tag === "ConfiguredRemote") return true;
    return Option.exists(
      slot.instance,
      (instance) => instance.lifecycle._tag === "Ready"
    );
  });

  return (
    <CliAppContent
      {...props}
      modelsConfigured={modelsConfigured}
      modelsReadyForInitialWork={modelsReadyForInitialWork}
      modelSetupActive={modelSetupActive}
      updateOnboardingResult={onboarding.updateResult}
    />
  );
}

function CliAppContent(
  props: CliAppProps & {
    readonly modelsConfigured: boolean;
    readonly modelsReadyForInitialWork: boolean;
    readonly modelSetupActive: boolean;
    readonly updateOnboardingResult: ReturnType<
      typeof useOnboardingState
    >["updateResult"];
  }
): ReactNode {
  useSessionPreload(!props.modelSetupActive);
  useFileWatchBridge();
  useSessionStartup({
    sessionStart: props.sessionStart,
    initialPrompt: props.initialPrompt,
    goal: props.goal,
    modelsConfigured:
      props.modelsReadyForInitialWork && !props.modelSetupActive,
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
  const onboardingSetup = useOnboardingModelSetup();
  const setupState = Option.getOrNull(Result.value(onboardingSetup.state));
  const modelSlotsState = Option.getOrNull(Result.value(useModelSlots()));
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
    Option.getOrElse(persistentNotificationStates, () => []),
  );
  const { rootSlotId } = useSlotProfiles();
  const localModelLoadActivity = modelSlotsState === null
    ? null
    : deriveLocalModelLoadActivity(modelSlotsState, rootSlotId);
  const describeError = (error: unknown): string => {
    if (error instanceof Error && error.message.length > 0) return error.message;
    if (typeof error === "object" && error !== null && "message" in error) {
      return String(error.message);
    }
    return "The local model setup could not be completed.";
  };
  const setupError = setupState?._tag === "Failed"
    ? (() => {
        const failure = setupState.failure;
        if (Schema.is(ModelDownloadFailureSchema)(failure)) {
          return modelDownloadFailureMessage(failure);
        }
        if (!("_tag" in failure)) return describeError(failure);
        switch (failure._tag) {
          case "OnboardingModelChoiceRejected":
            return "That model is no longer available for setup.";
          case "OnboardingModelResourceChanged":
            return "The selected model changed before setup completed. Choose it again to retry.";
          default:
            return describeError(failure);
        }
      })()
    : null;
  const cancelError = Result.matchWithError(onboardingSetup.cancelResult, {
    onInitial: () => null,
    onError: describeError,
    onDefect: (_, failure) => Cause.isInterruptedOnly(failure.cause)
      ? null
      : "The cancellation command failed unexpectedly.",
    onSuccess: () => null,
  });
  const setupOnboardingModel = onboardingSetup.setup;
  const cancelOnboardingModelSetup = onboardingSetup.cancel;
  const skipOnboardingModelSetup = onboardingSetup.skip;
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
    recentChatsEnabled: !props.modelSetupActive,
  });

  const setupPreparation = (
    progress: LocalModelsState["discoveryState"]["progress"],
    error: string | null
  ) => ({
    surface: (
      <OnboardingModelPreparation
        hardware={onboardingSetup.hardware}
        progress={progress}
        error={error}
        width={chatColumnWidth}
        onSkip={skipOnboardingModelSetup}
      />
    ),
    placeholder: "Preparing local models…",
  });
  const setupWithState = (state: OnboardingModelSetupState) => {
    if (state._tag === "Discovering") return setupPreparation(state.progress, null);
    if (state._tag === "DiscoveryFailed") {
      return setupPreparation(state.progress, state.failure.message);
    }
    const chooser = (
      operation: Parameters<typeof OnboardingModelChooser>[0]["operation"],
      placeholder: string,
    ) => ({
      surface: (
        <OnboardingModelChooser
          hardware={onboardingSetup.hardware}
          options={state.options}
          width={chatColumnWidth}
          error={setupError ?? cancelError}
          operation={operation}
          onSelect={setupOnboardingModel}
          onSkip={skipOnboardingModelSetup}
        />
      ),
      placeholder,
    });
    switch (state._tag) {
      case "Choosing": return chooser(null, "Select a model to start coding…");
      case "Preparing":
      case "Configuring": return chooser({
        _tag: "Configuring",
        model: state.model,
      }, `Configuring ${formatLocalModelDisplayName(state.model)}…`);
      case "Installing": return chooser({
        _tag: "Downloading",
        model: state.model,
        starting: state.model.acquisitionState._tag === "NotInstalled",
        cancelling: state.cancelling,
        cancelError,
        onCancel: cancelOnboardingModelSetup,
      }, `Downloading ${formatLocalModelDisplayName(state.model)}…`);
      case "Loading": return chooser({
        _tag: "Activating",
        providerModelId: state.providerModelId,
        displayName: formatLocalModelDisplayName(state.model),
        phase: state.phase,
        failure: state.failure,
        onRetry: () => setupOnboardingModel(state.configurationId),
        onChooseAnother: cancelOnboardingModelSetup,
      }, state.phase === "Loading"
        ? `Loading ${formatLocalModelDisplayName(state.model)}…`
        : state.phase === "Stopping"
          ? `Stopping ${formatLocalModelDisplayName(state.model)}…`
          : state.phase === "Ready"
            ? `Finishing setup for ${formatLocalModelDisplayName(state.model)}…`
            : `Couldn’t load ${formatLocalModelDisplayName(state.model)}`);
      case "Completing": return chooser({
        _tag: "Activating",
        providerModelId: state.providerModelId,
        displayName: formatLocalModelDisplayName(state.model),
        phase: "Ready",
        failure: null,
        onRetry: () => setupOnboardingModel(state.configurationId),
        onChooseAnother: cancelOnboardingModelSetup,
      }, `Finishing setup for ${formatLocalModelDisplayName(state.model)}…`);
      case "Failed": return chooser(null, "Select a model to start coding…");
    }
  };
  const setupPresentation = !props.modelSetupActive
    ? { surface: undefined, placeholder: null }
    : Result.match(onboardingSetup.state, {
        onInitial: () => setupPreparation([], null),
        onFailure: () =>
          setupPreparation([], "Local model discovery is unavailable."),
        onSuccess: ({ value }) => setupWithState(value),
      });
  const setupSurface = setupPresentation.surface;
  const modelSetupPlaceholder = setupPresentation.placeholder;
  const activityRail = (
    <ActivityRailContainer
      modelLoadActivity={localModelLoadActivity}
      onStopModel={slotActions.stop}
      width={chatColumnWidth}
      agentActivityEnabled={!props.modelSetupActive}
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
        !props.modelSetupActive &&
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
                  exclusiveEmptyState={props.modelSetupActive}
                />
                {!props.modelSetupActive && (
                  <box
                    style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}
                  >
                    <TaskListContainer />
                  </box>
                )}
                {props.modelSetupActive ? (
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
              {menu.open && !props.modelSetupActive ? (
                <ModelMenusContainer notificationState={notificationState} />
              ) : (
                <ComposerContainer
                  chatColumnWidth={chatColumnWidth}
                  clientWorkingDirectory={clientWorkingDirectory}
                  widgetNavActive={
                    widget.widgetNavActive && !props.modelSetupActive
                  }
                  handleWidgetKeyEvent={widget.navigation.handleKeyEvent}
                  modelsConfigured={props.modelsConfigured}
                  modelSetupInProgress={props.modelSetupActive}
                  modelSetupPlaceholder={
                    props.modelSetupActive ? modelSetupPlaceholder : null
                  }
                  notificationState={notificationState}
                />
              )}
            </box>
          </box>

          {clipboardToast && (
            <Toast
              color={theme.success}
              background={theme.surface}
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
