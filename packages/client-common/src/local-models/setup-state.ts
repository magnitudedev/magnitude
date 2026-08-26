import { Cause, Data, Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import type {
  LocalModel,
  LocalModelDiscoveryProgressStep,
  LocalModelsState,
  ModelDownloadFailure,
  ModelInstanceFailure,
  ModelResidency,
  ModelSlotsState,
  ProviderModelId,
  SlotSelection,
} from "@magnitudedev/sdk"
import type { OnboardingPersistenceError } from "../onboarding/persistence"
import type {
  HarnessConnectionError,
  HarnessDestination,
  HarnessId,
  HarnessLaunchPlan,
} from "../harness-connections/service"
import type {
  ModelSlotsAssignError,
  ModelSlotsLoadError,
  ModelSlotsStopError,
} from "../model-slots/service"
import { localModelOptions, type LocalModelOption } from "./options"
import type { LocalModelsCancelError, LocalModelsInstallError } from "./service"
import { findLocalModelById } from "./projection"

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
  readonly modelId: ProviderModelId
  readonly reason: "missing" | "unresolved" | "ineligible"
}> {}

export class OnboardingModelResourceChanged extends Data.TaggedError(
  "OnboardingModelResourceChanged",
)<{
  readonly modelId: ProviderModelId
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
  | HarnessConnectionError

export type OnboardingModelSetupNotice = {
  readonly failure: OnboardingModelSetupFailure
  readonly subject:
    | { readonly _tag: "Setup" }
    | {
        readonly _tag: "ModelOperation"
        readonly operation: OnboardingModelSetupExecution["_tag"]
        readonly model: LocalModel
      }
}

export type OnboardingModelSetupExitKind = "Skip" | "Close"

export interface OnboardingModelRankingControls {
  readonly fastToSmart: number
}

export const defaultOnboardingModelRankingControls: OnboardingModelRankingControls = {
  fastToSmart: 0.5,
}

const normalizeRankingControl = (value: number): number => Number.isFinite(value)
  ? Math.min(1, Math.max(0, value))
  : 0

export const normalizeOnboardingModelRankingControls = (
  controls: OnboardingModelRankingControls,
): OnboardingModelRankingControls => ({
  fastToSmart: normalizeRankingControl(controls.fastToSmart),
})

export type OnboardingModelLoadStatus =
  | { readonly _tag: "Preparing" }
  | {
      readonly _tag: "Loading"
      readonly stage: Extract<ModelResidency, { readonly _tag: "Loading" }>["stage"]
      readonly progress: Option.Option<number>
    }
  | { readonly _tag: "Cancelling" }
  | { readonly _tag: "Stopping" }
  | { readonly _tag: "Ready" }
  | { readonly _tag: "Failed"; readonly failure: ModelInstanceFailure }

export type OnboardingModelSetupOperation =
  | {
      readonly _tag: "Preparing" | "Installing" | "Configuring"
      readonly model: LocalModel
      readonly cancelling: boolean
    }
  | {
      readonly _tag: "Loading"
      readonly model: LocalModel
      readonly modelId: ProviderModelId
      readonly providerModelId: ProviderModelId
      readonly status: OnboardingModelLoadStatus
    }
  | {
      readonly _tag: "Completing"
      readonly model: LocalModel
      readonly modelId: ProviderModelId
      readonly providerModelId: ProviderModelId
    }

export type OnboardingModelSetupContent =
  | {
      readonly _tag: "Preparation"
      readonly progress: readonly LocalModelDiscoveryProgressStep[]
      readonly discoveryFailure: Extract<
        LocalModelsState["discoveryState"],
        { readonly _tag: "Failed" }
      >["failure"] | null
    }
  | {
      readonly _tag: "Chooser"
      readonly options: readonly LocalModelOption[]
      readonly rankingControls: OnboardingModelRankingControls
      readonly operation: Option.Option<OnboardingModelSetupOperation>
    }
  | {
      readonly _tag: "Harness"
      readonly model: LocalModel
      readonly modelId: ProviderModelId
      readonly providerModelId: ProviderModelId
      readonly destinations: ReadonlyArray<HarnessDestination>
    }
  | {
      readonly _tag: "ApplyingHarness"
      readonly model: LocalModel
      readonly modelId: ProviderModelId
      readonly harness: HarnessId
    }
  | {
      readonly _tag: "HarnessHandoff"
      readonly plan: HarnessLaunchPlan
    }
  | { readonly _tag: "Closing" }

export type OnboardingModelSetupState =
  | { readonly _tag: "Closed" }
  | {
      readonly _tag: "Open"
      readonly exitKind: OnboardingModelSetupExitKind
      readonly notice: Option.Option<OnboardingModelSetupNotice>
      readonly content: OnboardingModelSetupContent
    }

export type OnboardingModelSetupExecution =
  | {
      readonly _tag: "Preparing" | "Installing" | "Configuring"
      readonly option: LocalModelOption
      readonly modelId: ProviderModelId
      readonly cancelling: boolean
    }
  | {
      readonly _tag: "Loading"
      readonly option: LocalModelOption
      readonly modelId: ProviderModelId
      readonly providerModelId: ProviderModelId
      readonly selection: SlotSelection
      readonly cancelling: boolean
    }
  | {
      readonly _tag: "Completing"
      readonly option: LocalModelOption
      readonly modelId: ProviderModelId
      readonly providerModelId: ProviderModelId
    }

export type OnboardingModelSetupAttempt =
  | OnboardingModelSetupExecution
  | {
      readonly _tag: "LoadFailed"
      readonly execution: Extract<OnboardingModelSetupExecution, { readonly _tag: "Loading" }>
      readonly failure: ModelInstanceFailure
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
  attempt: Option.Option<OnboardingModelSetupAttempt>,
  models: LocalModelsState,
  slots: ModelSlotsState,
  rankingControls: OnboardingModelRankingControls,
): OnboardingModelSetupContent => {
  if (Option.isNone(attempt)) {
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
      rankingControls,
      operation: Option.none(),
    }
  }

  const current = attempt.value
  const execution = current._tag === "LoadFailed" ? current.execution : current
  const options = localModelOptions(models, slots)
  const currentModel = Option.getOrElse(
    findLocalModelById(
      models.models,
      execution.modelId,
    ),
    () => execution.option.model,
  )

  if (current._tag === "LoadFailed") {
    return {
      _tag: "Chooser",
      options,
      rankingControls,
      operation: Option.some({
        _tag: "Loading",
        model: currentModel,
        modelId: current.execution.modelId,
        providerModelId: current.execution.providerModelId,
        status: { _tag: "Failed", failure: current.failure },
      }),
    }
  }

  if (execution._tag === "Completing") {
    return {
      _tag: "Chooser",
      options,
      rankingControls,
      operation: Option.some({
        _tag: "Completing",
        model: currentModel,
        modelId: execution.modelId,
        providerModelId: execution.providerModelId,
      }),
    }
  }
  if (execution._tag !== "Loading") {
    return {
      _tag: "Chooser",
      options,
      rankingControls,
      operation: Option.some({
        _tag: execution._tag,
        model: currentModel,
        cancelling: execution.cancelling,
      }),
    }
  }

  const slot = slots.slots.primary
  const residency = slot._tag === "ConfiguredLocal"
    && sameSelection(slot.selection, execution.selection)
    ? Option.some(slot.residency)
    : Option.none()
  const status: OnboardingModelLoadStatus = execution.cancelling
    ? { _tag: "Cancelling" }
    : Option.match(residency, {
        onNone: () => ({ _tag: "Preparing" }),
        onSome: (value) => {
          switch (value._tag) {
            // A failed residency becomes interactive only after the selection
            // fiber has retained it as a LoadFailed operation result. Keeping
            // the in-flight projection non-terminal avoids exposing retry
            // controls before that transition is complete.
            case "Failed": return { _tag: "Preparing" as const }
            case "Stopping":
            case "Unloaded": return { _tag: "Stopping" as const }
            case "Ready": return { _tag: "Ready" as const }
            case "Requested": return { _tag: "Preparing" as const }
            case "Loading": return {
              _tag: "Loading" as const,
              stage: value.stage,
              progress: value.progress,
            }
          }
        },
      })
  return {
    _tag: "Chooser",
    options,
    rankingControls,
    operation: Option.some({
      _tag: "Loading",
      model: currentModel,
      modelId: execution.modelId,
      providerModelId: execution.providerModelId,
      status,
    }),
  }
}
