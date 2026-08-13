import { Data, Option } from "effect"
import type {
  LocalModel,
  ModelDownloadFailure,
  LocalModelRecommendationProgressStep,
  LocalModelsState,
  ModelInstanceFailure,
  ModelInstanceId,
  ModelSlotInstanceLifecycle,
  ModelServingConfigurationId,
  ModelSlotsState,
  ProviderModelId,
} from "@magnitudedev/sdk"
import type { OnboardingPersistenceError } from "../onboarding/persistence"
import type {
  ModelSlotsAssignError,
  ModelSlotsLoadError,
  ModelSlotsStopError,
} from "../model-slots/service"
import { localModelOptions, type LocalModelOption } from "./options"
import type { LocalModelsCancelError, LocalModelsInstallError } from "./service"
import {
  findLocalModelByConfigurationId,
  localModelProviderModelId,
} from "./projection"

export class OnboardingModelSetupAlreadyActive extends Data.TaggedError(
  "OnboardingModelSetupAlreadyActive",
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
  readonly reason: "missing" | "unresolved" | "ineligible" | "missing_provider_identity"
}> {}

export class OnboardingModelResourceChanged extends Data.TaggedError(
  "OnboardingModelResourceChanged",
)<{
  readonly configurationId: ModelServingConfigurationId
  readonly resource: "installation" | "instance"
}> {}

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

type WithOptions<State> = State & { readonly options: readonly LocalModelOption[] }

export type OnboardingModelSetupState =
  | {
      readonly _tag: "Discovering"
      readonly progress: readonly LocalModelRecommendationProgressStep[]
    }
  | {
      readonly _tag: "DiscoveryFailed"
      readonly progress: readonly LocalModelRecommendationProgressStep[]
      readonly failure: Extract<LocalModelsState["discoveryState"], { readonly _tag: "Failed" }>["failure"]
    }
  | WithOptions<{ readonly _tag: "Choosing" }>
  | WithOptions<{
      readonly _tag: "Preparing" | "Configuring"
      readonly model: LocalModel
      readonly cancelling: boolean
    }>
  | WithOptions<{
      readonly _tag: "Installing"
      readonly model: LocalModel
      readonly cancelling: boolean
    }>
  | WithOptions<{
      readonly _tag: "Loading"
      readonly configurationId: ModelServingConfigurationId
      readonly providerModelId: ProviderModelId
      readonly model: LocalModel
      readonly phase: "Loading" | "Stopping" | "Ready" | "Failed"
      readonly failure: ModelInstanceFailure | null
    }>
  | WithOptions<{
      readonly _tag: "Completing"
      readonly configurationId: ModelServingConfigurationId
      readonly providerModelId: ProviderModelId
      readonly model: LocalModel
    }>
  | WithOptions<{
      readonly _tag: "Failed"
      readonly configurationId: ModelServingConfigurationId
      readonly model: Option.Option<LocalModel>
      readonly failure: OnboardingModelSetupFailure
    }>

export type OnboardingModelSetupExecution =
  | { readonly _tag: "Choosing" }
  | {
      readonly _tag: "Preparing" | "Installing" | "Configuring"
      readonly configurationId: ModelServingConfigurationId
      readonly cancelling: boolean
    }
  | {
      readonly _tag: "Loading"
      readonly configurationId: ModelServingConfigurationId
      readonly instanceId: ModelInstanceId
      readonly cancelling: boolean
    }
  | {
      readonly _tag: "Completing"
      readonly configurationId: ModelServingConfigurationId
    }
  | {
      readonly _tag: "Failed"
      readonly configurationId: ModelServingConfigurationId
      readonly failure: OnboardingModelSetupFailure
    }

export const projectOnboardingModelSetup = (
  execution: OnboardingModelSetupExecution,
  models: LocalModelsState,
  slots: ModelSlotsState,
): OnboardingModelSetupState => {
  if (models.discoveryState._tag === "Loading") {
    return { _tag: "Discovering", progress: models.discoveryState.progress }
  }
  if (models.discoveryState._tag === "Failed") {
    return {
      _tag: "DiscoveryFailed",
      progress: models.discoveryState.progress,
      failure: models.discoveryState.failure,
    }
  }
  const options = localModelOptions(models, slots)
  if (execution._tag === "Choosing") return { _tag: "Choosing", options }
  const model = findLocalModelByConfigurationId(models.models, execution.configurationId)
  if (execution._tag === "Failed") return { ...execution, model, options }
  if (Option.isNone(model)) {
    return {
      _tag: "Failed",
      configurationId: execution.configurationId,
      model,
      options,
      failure: new OnboardingModelResourceChanged({
        configurationId: execution.configurationId,
        resource: execution._tag === "Loading" ? "instance" : "installation",
      }),
    }
  }
  if (execution._tag !== "Loading" && execution._tag !== "Completing") {
    return { ...execution, model: model.value, options }
  }
  const providerModelId = localModelProviderModelId(model.value)
  if (Option.isNone(providerModelId)) {
    return {
      _tag: "Failed",
      configurationId: execution.configurationId,
      model,
      options,
      failure: new OnboardingModelResourceChanged({
        configurationId: execution.configurationId,
        resource: "instance",
      }),
    }
  }
  if (execution._tag === "Completing") {
    return {
      _tag: "Completing",
      configurationId: execution.configurationId,
      providerModelId: providerModelId.value,
      model: model.value,
      options,
    }
  }
  const slot = slots.slots.primary
  const lifecycle = slot._tag === "ConfiguredLocal"
    ? Option.flatMap(slot.instance, (instance) =>
        instance.id === execution.instanceId
          && instance.configurationId === execution.configurationId
          ? Option.some(instance.lifecycle)
          : Option.none())
    : Option.none<ModelSlotInstanceLifecycle>()
  const phase = execution.cancelling
    ? "Stopping" as const
    : Option.match(lifecycle, {
        onNone: () => "Loading" as const,
        onSome: (value) => value._tag === "Failed" ? "Failed" as const
          : value._tag === "Stopping" || value._tag === "Stopped" ? "Stopping" as const
          : value._tag === "Ready" ? "Ready" as const
          : "Loading" as const,
      })
  return {
    _tag: "Loading",
    configurationId: execution.configurationId,
    providerModelId: providerModelId.value,
    model: model.value,
    phase,
    failure: Option.match(lifecycle, {
      onNone: () => null,
      onSome: (value) => value._tag === "Failed" ? value.failure : null,
    }),
    options,
  }
}
