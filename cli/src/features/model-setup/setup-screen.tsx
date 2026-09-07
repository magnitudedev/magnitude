import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { useKeyboard, useTerminalDimensions } from "@opentui/react"
import {
  useOnboardingModelSetup, formatLocalModelDisplayName, onboardingModelSetupNoticeMessage,
  type OnboardingModelSetupState,
} from "@magnitudedev/client-common"
import type { LocalModelPreparation } from "@magnitudedev/sdk"
import { useTheme } from "../../hooks/use-theme"
import { OnboardingModelChooser, onboardingSetupAdditionalRows, OnboardingModelExiting, OnboardingModelPreparation, HarnessChooser, SetupFrame } from "./index"

const unresolvedLocalModelPreparation: LocalModelPreparation = {
  discovery: { complete: false, modelsFound: 0 },
  assessment: { complete: false, settledModels: 0, totalModels: 0 },
};

/** The same presentation for ordinary onboarding and a terminal-hosted setup. */
export function useSetupPresentation(
  onboardingSetup: ReturnType<typeof useOnboardingModelSetup>,
  chatColumnWidth: number,
  active = true,
) {
  const theme = useTheme();
  const setupOnboardingModel = onboardingSetup.select;
  const cancelOnboardingModelSetup = onboardingSetup.cancel;
  const chooseAnotherOnboardingModel = onboardingSetup.chooseAnother;
  const setupAdditionalRows = onboardingSetupAdditionalRows(onboardingSetup.hardware, chatColumnWidth);
  const setupPreparation = (
    preparation: LocalModelPreparation = unresolvedLocalModelPreparation,
    error: string | null = null,
  ) => ({
    surface: (
      <OnboardingModelPreparation
        hardware={onboardingSetup.hardware}
        preparation={preparation}
        error={error}
        width={chatColumnWidth}
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
      return setupPreparation(state.content.preparation);
    }
    if (state.content._tag === "Harness") {
      return {
        surface: (
          <HarnessChooser
            width={chatColumnWidth}
            additionalRows={setupAdditionalRows}
            model={state.content.model}
            destinations={state.content.destinations}
            applying={null}
            onContinue={onboardingSetup.continueWithHarness}
          />
        ),
        placeholder: null,
      };
    }
    if (state.content._tag === "ApplyingHarness") {
      return {
        surface: (
          <SetupFrame width={chatColumnWidth} stage="harness" additionalRows={setupAdditionalRows}>
            <text style={{ fg: theme.accent }}>Connecting {state.content.harness}…</text>
            <text style={{ fg: theme.text.supporting }}>Configuring the selected harness for {formatLocalModelDisplayName(state.content.model)}</text>
          </SetupFrame>
        ),
        placeholder: null,
      };
    }
    if (state.content._tag === "ReturnToHost") {
      return { surface: <SetupFrame width={chatColumnWidth} stage="harness"><text>Returning to Pi…</text></SetupFrame>, placeholder: null };
    }
    if (state.content._tag === "HarnessHandoff") {
      return {
        surface: (
          <SetupFrame width={chatColumnWidth} stage="harness" additionalRows={setupAdditionalRows}>
            <text style={{ fg: theme.accent }}>Launching {state.content.plan.harness}…</text>
          </SetupFrame>
        ),
        placeholder: null,
      };
    }
    const content = state.content;
    const setupError = Option.match(state.notice, {
      onNone: () => null,
      onSome: onboardingModelSetupNoticeMessage,
    });
    const chooser = (
      operation: Parameters<typeof OnboardingModelChooser>[0]["operation"],
      placeholder: string,
    ) => ({
      surface: (
        <OnboardingModelChooser
          hardware={onboardingSetup.hardware}
          options={content.options}
          rankingControls={content.rankingControls}
          onRankingControlsChange={onboardingSetup.setRankingControls}
          width={chatColumnWidth}
          error={setupError}
          operation={operation}
          onSelect={setupOnboardingModel}
        />
      ),
      placeholder,
    });
    return Option.match(content.operation, {
      onNone: () => chooser(null, "Select a model to start coding…"),
      onSome: (operation) => {
        switch (operation._tag) {
          case "Preparing": {
            const acquisition = operation.model._tag === "Catalog"
              ? operation.model.acquisitionState
              : undefined;
            if (acquisition !== undefined && acquisition._tag !== "Installed") {
              const starting = acquisition._tag !== "Installing";
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
            starting: operation.model._tag === "Catalog"
              && operation.model.acquisitionState._tag === "NotInstalled",
            cancelling: operation.cancelling,
            onCancel: cancelOnboardingModelSetup,
          }, `Downloading ${formatLocalModelDisplayName(operation.model)}…`);
          case "Loading": return chooser({
            _tag: "Activating",
            providerModelId: operation.providerModelId,
            model: operation.model,
            status: operation.status,
            onCancel: cancelOnboardingModelSetup,
            onRetry: () => setupOnboardingModel(operation.modelId),
            onChooseAnother: chooseAnotherOnboardingModel,
          }, operation.status._tag === "Preparing" || operation.status._tag === "Loading"
            ? `Loading ${formatLocalModelDisplayName(operation.model)}…`
            : operation.status._tag === "Cancelling"
              ? `Cancelling loading for ${formatLocalModelDisplayName(operation.model)}…`
              : operation.status._tag === "Stopping"
                ? `Stopping ${formatLocalModelDisplayName(operation.model)}…`
                : operation.status._tag === "Ready"
                  ? `Finishing setup for ${formatLocalModelDisplayName(operation.model)}…`
                  : `Couldn’t load ${formatLocalModelDisplayName(operation.model)}`);
          case "Completing": return chooser({
            _tag: "Activating",
            providerModelId: operation.providerModelId,
            model: operation.model,
            status: { _tag: "Ready" },
            onCancel: cancelOnboardingModelSetup,
            onRetry: () => setupOnboardingModel(operation.modelId),
            onChooseAnother: chooseAnotherOnboardingModel,
          }, `Finishing setup for ${formatLocalModelDisplayName(operation.model)}…`);
        }
      },
    });
  };
  const setupPresentation = !active
    ? { surface: undefined, placeholder: null }
    : Result.match(onboardingSetup.view, {
        onInitial: () => setupPreparation(),
        onFailure: () => setupPreparation(
          unresolvedLocalModelPreparation,
          "Local model setup is unavailable.",
        ),
        onSuccess: ({ value }) => setupWithState(value),
      });

  return setupPresentation;
}

export function HostedSetupScreen() {
  const setup = useOnboardingModelSetup();
  const { width } = useTerminalDimensions();
  const presentation = useSetupPresentation(setup, width - 2);
  useKeyboard((key) => {
    if (key.ctrl && key.name === "c" && !key.meta && !key.option) process.kill(process.pid, "SIGINT");
  });
  return <box style={{ width: "100%", height: "100%", flexDirection: "column", padding: 1 }}>{presentation.surface}</box>;
}
