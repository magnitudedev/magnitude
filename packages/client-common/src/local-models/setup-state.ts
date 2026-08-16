import { Cause, Data, Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import type {
  LocalModel,
  LocalModelRecommendationProgressStep,
  LocalModelsState,
  ModelDownloadFailure,
  ModelInstanceFailure,
  ModelServingConfigurationId,
  ModelSlotsState,
  ProviderModelId,
  SlotSelection,
} from "@magnitudedev/sdk"
import type { OnboardingPersistenceError } from "../onboarding/persistence"
import type {
  ModelSlotsAssignError,
  ModelSlotsLoadError,
  ModelSlotsStopError,
} from "../model-slots/service"
import { localModelOptions, type LocalModelOption } from "./options"
import type { LocalModelsCancelError, LocalModelsInstallError } from "./service"
import { findLocalModelByConfigurationId } from "./projection"

export class OnboardingModelSetupAlreadyActive extends Data.TaggedError(
  "OnboardingModelSetupAlreadyActive",
)<{}> {}

export class OnboardingModelSetupNotOpen extends Data.TaggedError(
  "OnboardingModelSetupNotOpen",
)<{}> {}

export class OnboardingModelSetupNotActive extends Data.TaggedError(
  "OnboardingModelSetupNotActive",
)<{}> {}

export class OnboardingModelSetupCancellationUnavailable extends Data.TaggedError(
  "OnboardingModelSetupCancellationUnavailable",
)<{}> {}

export class OnboardingModelChoiceRejected extends Data.TaggedError(
  "OnboardingModelChoiceRejected",
)<{
  readonly configurationId: ModelServingConfigurationId
  readonly reason: "missing" | "unresolved" | "ineligible"
}> {}

export class OnboardingModelResourceChanged extends Data.TaggedError(
  "OnboardingModelResourceChanged",
)<{
  readonly configurationId: ModelServingConfigurationId
  readonly resource: "installation" | "instance"
}> {}

export class OnboardingModelSetupObservationFailed extends Data.TaggedError(
  "OnboardingModelSetupObservationFailed",
)<{ readonly source: "onboarding" | "local-models" | "model-slots" }> {}

export type OnboardingModelSetupFailure =
  | OnboardingModelChoiceRejected
  | OnboardingModelResourceChanged
  | ModelDownloadFailure
  | ModelInstanceFailure
  | LocalModelsInstallError
  | LocalModelsCancelError
  | ModelSlotsAssignError
  | ModelSlotsLoadError
  | ModelSlotsStopError
  | OnboardingPersistenceError

export type OnboardingModelSetupExitKind = "Skip" | "Close"

export type OnboardingModelSetupOperation =
  | {
      readonly _tag: "Preparing" | "Installing" | "Configuring"
      readonly model: LocalModel
      readonly cancelling: boolean
    }
  | {
      readonly _tag: "Loading"
      readonly model: LocalModel
      readonly configurationId: ModelServingConfigurationId
      readonly providerModelId: ProviderModelId
      readonly phase: "Loading" | "Stopping" | "Ready" | "Failed"
      readonly failure: ModelInstanceFailure | null
    }
  | {
      readonly _tag: "Completing"
      readonly model: LocalModel
      readonly configurationId: ModelServingConfigurationId
      readonly providerModelId: ProviderModelId
    }

export type OnboardingModelSetupContent =
  | {
      readonly _tag: "Preparation"
      readonly progress: readonly LocalModelRecommendationProgressStep[]
      readonly discoveryFailure: Extract<
        LocalModelsState["discoveryState"],
        { readonly _tag: "Failed" }
      >["failure"] | null
    }
  | {
      readonly _tag: "Chooser"
      readonly options: readonly LocalModelOption[]
      readonly operation: Option.Option<OnboardingModelSetupOperation>
    }
  | { readonly _tag: "Closing" }

export type OnboardingModelSetupState =
  | { readonly _tag: "Closed" }
  | {
      readonly _tag: "Open"
      readonly exitKind: OnboardingModelSetupExitKind
      readonly notice: Option.Option<OnboardingModelSetupFailure>
      readonly content: OnboardingModelSetupContent
    }

export type OnboardingModelSetupExecution =
  | {
      readonly _tag: "Preparing" | "Installing" | "Configuring"
      readonly option: LocalModelOption
      readonly configurationId: ModelServingConfigurationId
      readonly cancelling: boolean
    }
  | {
      readonly _tag: "Loading"
      readonly option: LocalModelOption
      readonly configurationId: ModelServingConfigurationId
      readonly providerModelId: ProviderModelId
      readonly selection: SlotSelection
      readonly cancelling: boolean
    }
  | {
      readonly _tag: "Completing"
      readonly option: LocalModelOption
      readonly configurationId: ModelServingConfigurationId
      readonly providerModelId: ProviderModelId
    }

export const tagOnboardingModelSetupObservation = <Value, Error>(
  result: Result.Result<Value, Error>,
  source: OnboardingModelSetupObservationFailed["source"],
): Result.Result<Value, OnboardingModelSetupObservationFailed> => {
  if (Result.isInitial(result)) {
    return Result.initial<Value, OnboardingModelSetupObservationFailed>(result.waiting)
  }
  if (Result.isSuccess(result)) {
    return Result.success(result.value, { waiting: result.waiting })
  }
  return Result.failure(
    Cause.map(result.cause, () => new OnboardingModelSetupObservationFailed({ source })),
    {
      previousSuccess: Option.map(result.previousSuccess, (previous) =>
        Result.success<Value, OnboardingModelSetupObservationFailed>(previous.value, {
          waiting: previous.waiting,
        })),
      waiting: result.waiting,
    },
  )
}

const sameSelection = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId
  && left.reasoningEffort === right.reasoningEffort

export const projectOnboardingModelSetupContent = (
  execution: Option.Option<OnboardingModelSetupExecution>,
  models: LocalModelsState,
  slots: ModelSlotsState,
): OnboardingModelSetupContent => {
  if (Option.isNone(execution)) {
    if (models.discoveryState._tag === "Loading") {
      return {
        _tag: "Preparation",
        progress: models.discoveryState.progress,
        discoveryFailure: null,
      }
    }
    if (models.discoveryState._tag === "Failed") {
      return {
        _tag: "Preparation",
        progress: models.discoveryState.progress,
        discoveryFailure: models.discoveryState.failure,
      }
    }
    return {
      _tag: "Chooser",
      options: localModelOptions(models, slots),
      operation: Option.none(),
    }
  }

  const current = execution.value
  const options = localModelOptions(models, slots)
  const currentModel = Option.getOrElse(
    findLocalModelByConfigurationId(
      models.models,
      current.configurationId,
    ),
    () => current.option.model,
  )

  if (current._tag === "Completing") {
    return {
      _tag: "Chooser",
      options,
      operation: Option.some({
        _tag: "Completing",
        model: currentModel,
        configurationId: current.configurationId,
        providerModelId: current.providerModelId,
      }),
    }
  }
  if (current._tag !== "Loading") {
    return {
      _tag: "Chooser",
      options,
      operation: Option.some({
        _tag: current._tag,
        model: currentModel,
        cancelling: current.cancelling,
      }),
    }
  }

  const slot = slots.slots.primary
  const residency = slot._tag === "ConfiguredLocal"
    && sameSelection(slot.selection, current.selection)
    && (slot.residency._tag !== "Loading"
      && slot.residency._tag !== "Ready"
      && slot.residency._tag !== "Stopping"
      || slot.residency.configurationId === current.configurationId)
    ? Option.some(slot.residency)
    : Option.none()
  const failure = Option.getOrNull(Option.flatMap(
    residency,
    (value) => value._tag === "Failed" ? Option.some(value.failure) : Option.none(),
  ))
  const phase = current.cancelling
    ? "Stopping" as const
    : Option.match(residency, {
        onNone: () => "Loading" as const,
        onSome: (value) => {
          switch (value._tag) {
            case "Failed": return "Failed" as const
            case "Stopping":
            case "Unloaded": return "Stopping" as const
            case "Ready": return "Ready" as const
            case "Requested":
            case "Loading": return "Loading" as const
          }
        },
      })
  return {
    _tag: "Chooser",
    options,
    operation: Option.some({
      _tag: "Loading",
      model: currentModel,
      configurationId: current.configurationId,
      providerModelId: current.providerModelId,
      phase,
      failure,
    }),
  }
}
