import {
  formatLocalModelDisplayName,
  localModelServingProfile,
  localModelServingState,
  modelDownloadFailureMessage,
} from "@magnitudedev/client-common"
import {
  CatalogFormModelIdSchema,
  HuggingFaceFormModelIdSchema,
  ModelIdSchema,
  type LocalModel,
  type ModelTransferProgress,
} from "@magnitudedev/sdk"
import { Option, Schema } from "effect"

const NonNegativeSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.nonNegative(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)
const PositiveSafeInteger = NonNegativeSafeInteger.pipe(Schema.positive())
const Fraction = Schema.Number.pipe(Schema.finite(), Schema.between(0, 1))

const JsonFailureSchema = Schema.Struct({
  message: Schema.String,
})

const JsonTransferProgressSchema = Schema.Struct({
  stage: Schema.Literal("queued", "resolving", "checking_space", "downloading", "verifying", "publishing"),
  completedBytes: NonNegativeSafeInteger,
  totalBytes: NonNegativeSafeInteger,
  bytesPerSecond: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
})

const JsonCatalogInactiveInstallationSchema = Schema.Union(
  Schema.Struct({ state: Schema.Literal("not_installed") }),
  Schema.Struct({ state: Schema.Literal("installing"), progress: JsonTransferProgressSchema }),
  Schema.Struct({ state: Schema.Literal("install_failed"), error: JsonFailureSchema }),
)

const JsonResidentInstallationSchema = Schema.Union(
  Schema.Struct({ state: Schema.Literal("installed") }),
  Schema.Struct({ state: Schema.Literal("update_available") }),
  Schema.Struct({ state: Schema.Literal("updating"), progress: JsonTransferProgressSchema }),
  Schema.Struct({ state: Schema.Literal("update_failed"), error: JsonFailureSchema }),
  Schema.Struct({ state: Schema.Literal("removing") }),
  Schema.Struct({ state: Schema.Literal("remove_failed"), error: JsonFailureSchema }),
)

const JsonUnavailableInstallationSchema =
  Schema.Struct({ state: Schema.Literal("unavailable"), error: JsonFailureSchema })

const JsonInstallationSchema = Schema.Union(
  JsonCatalogInactiveInstallationSchema,
  JsonResidentInstallationSchema,
  JsonUnavailableInstallationSchema,
)

const JsonResidencySchema = Schema.Union(
  Schema.Struct({ state: Schema.Literal("unloaded") }),
  Schema.Struct({ state: Schema.Literal("requested") }),
  Schema.Struct({
    state: Schema.Literal("loading"),
    stage: Schema.Literal("queued", "resolving", "unloading", "loading", "verifying"),
    progress: Schema.optionalWith(Fraction, { as: "Option", exact: true }),
  }),
  Schema.Struct({ state: Schema.Literal("ready") }),
  Schema.Struct({
    state: Schema.Literal("stopping"),
    reason: Schema.Literal("user_stop", "idle_timeout", "replacement", "memory_pressure"),
  }),
  Schema.Struct({
    state: Schema.Literal("failed"),
    error: Schema.Struct({
      message: Schema.String,
      retryable: Schema.Boolean,
    }),
  }),
)

const JsonLocalModelPresentationFields = {
  displayName: Schema.String.pipe(Schema.minLength(1)),
} as const

const JsonAssessedModelFields = {
  memoryBytes: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  contextLength: Schema.optionalWith(PositiveSafeInteger, { as: "Option", exact: true }),
} as const

export const JsonLocalModelSchema = Schema.Union(
  Schema.Struct({
    ...JsonLocalModelPresentationFields,
    ...JsonAssessedModelFields,
    modelId: CatalogFormModelIdSchema,
    source: Schema.Literal("catalog"),
    installation: JsonCatalogInactiveInstallationSchema,
  }),
  Schema.Struct({
    ...JsonLocalModelPresentationFields,
    ...JsonAssessedModelFields,
    modelId: CatalogFormModelIdSchema,
    source: Schema.Literal("catalog"),
    installation: JsonResidentInstallationSchema,
    residency: JsonResidencySchema,
  }),
  Schema.Struct({
    ...JsonLocalModelPresentationFields,
    modelId: HuggingFaceFormModelIdSchema,
    source: Schema.Literal("discovered"),
    installation: JsonUnavailableInstallationSchema,
  }),
  Schema.Struct({
    ...JsonLocalModelPresentationFields,
    ...JsonAssessedModelFields,
    modelId: HuggingFaceFormModelIdSchema,
    source: Schema.Literal("discovered"),
    installation: Schema.Struct({ state: Schema.Literal("installed") }),
    residency: JsonResidencySchema,
  }),
)
export type JsonLocalModel = typeof JsonLocalModelSchema.Type

const JsonModelsStatusViewSchema = Schema.Literal("list", "detail")

export const ModelsStatusJsonDataSchema = Schema.Union(
  Schema.Struct({
    state: Schema.Literal("initializing"),
    view: JsonModelsStatusViewSchema,
  }),
  Schema.Struct({
    state: Schema.Literal("ready"),
    view: Schema.Literal("list"),
    models: Schema.Array(JsonLocalModelSchema),
  }),
  Schema.Struct({
    state: Schema.Literal("ready"),
    view: Schema.Literal("detail"),
    model: JsonLocalModelSchema,
  }),
)
export type ModelsStatusJsonData = typeof ModelsStatusJsonDataSchema.Type

export const ModelsLoadJsonDataSchema = Schema.Struct({
  modelId: ModelIdSchema,
  outcome: Schema.Literal("load_requested"),
})

export const ModelsStopJsonDataSchema = Schema.Struct({
  outcome: Schema.Literal("stopped"),
})

export const modelsLoadJsonData = (modelId: typeof ModelIdSchema.Type) => ({
  modelId,
  outcome: "load_requested" as const,
})

export const modelsStopJsonData = () => ({ outcome: "stopped" as const })

export type ModelsStatusResult =
  | { readonly _tag: "Initializing"; readonly view: "list" | "detail" }
  | { readonly _tag: "List"; readonly models: readonly LocalModel[] }
  | { readonly _tag: "Detail"; readonly model: LocalModel }

export const modelsForStatus = (models: readonly LocalModel[]): LocalModel[] => models
  .filter((model) => model._tag === "Discovered" || model.acquisitionState._tag !== "NotInstalled")
  .sort((left, right) => formatLocalModelDisplayName(left).localeCompare(formatLocalModelDisplayName(right))
    || left.modelId.localeCompare(right.modelId))

const modelMemoryBytes = (model: LocalModel): Option.Option<number> => Option.flatMap(
  localModelServingState(model),
  (serving) => serving._tag !== "Assessed"
    ? Option.none()
    : serving.assessment._tag === "Fits"
      ? Option.some(serving.assessment.memory.totalRequiredBytes)
      : serving.assessment._tag === "DoesNotFit"
        ? Option.some(serving.assessment.totalRequiredBytes)
        : Option.none(),
)

const transferProgress = (progress: ModelTransferProgress): typeof JsonTransferProgressSchema.Encoded => ({
  stage: progress.stage,
  completedBytes: progress.completedBytes,
  totalBytes: progress.totalBytes,
  ...Option.match(progress.bytesPerSecond, {
    onNone: () => ({}),
    onSome: (bytesPerSecond) => ({ bytesPerSecond }),
  }),
})

const installation = (model: LocalModel): typeof JsonInstallationSchema.Encoded => {
  if (model._tag === "Discovered") {
    return model.state._tag === "Ready"
      ? { state: "installed" }
      : { state: "unavailable", error: { message: model.state.failure.message } }
  }

  const state = model.acquisitionState
  switch (state._tag) {
    case "NotInstalled": return { state: "not_installed" }
    case "Installing": return { state: "installing", progress: transferProgress(state.progress) }
    case "InstallFailed": return {
      state: "install_failed",
      error: { message: modelDownloadFailureMessage(state.failure) },
    }
    case "Installed": return { state: "installed" }
    case "UpdateAvailable": return { state: "update_available" }
    case "Updating": return { state: "updating", progress: transferProgress(state.progress) }
    case "UpdateFailed": return {
      state: "update_failed",
      error: { message: modelDownloadFailureMessage(state.failure) },
    }
    case "Removing": return { state: "removing" }
    case "RemoveFailed": return {
      state: "remove_failed",
      error: { message: state.failure.message },
    }
  }
}

const residency = (model: LocalModel): Option.Option<typeof JsonResidencySchema.Encoded> => {
  const state = model._tag === "Discovered"
    ? model.state._tag === "Ready" ? model.state.residencyState : undefined
    : "residencyState" in model.acquisitionState ? model.acquisitionState.residencyState : undefined
  return Option.fromNullable(state).pipe(
    Option.map((value) => {
      switch (value._tag) {
        case "Unloaded": return { state: "unloaded" as const }
        case "Requested": return { state: "requested" as const }
        case "Loading": return {
          state: "loading" as const,
          stage: value.stage,
          ...Option.match(value.progress, {
            onNone: () => ({}),
            onSome: (progress) => ({ progress }),
          }),
        }
        case "Ready": return { state: "ready" as const }
        case "Stopping": return { state: "stopping" as const, reason: value.reason }
        case "Failed": return {
          state: "failed" as const,
          error: {
            message: value.failure.message,
            retryable: value.failure.retryable,
          },
        }
      }
    }),
  )
}

export const localModelJson = (model: LocalModel): JsonLocalModel => {
  const projectedResidency = Option.getOrUndefined(residency(model))
  const projectedMemory = Option.getOrUndefined(modelMemoryBytes(model))
  const projectedContext = Option.getOrUndefined(
    Option.map(localModelServingProfile(model), ({ contextLength }) => contextLength),
  )
  return Schema.decodeUnknownSync(JsonLocalModelSchema)({
    modelId: model.modelId,
    displayName: formatLocalModelDisplayName(model),
    source: model._tag === "Catalog" ? "catalog" : "discovered",
    ...(projectedMemory === undefined ? {} : { memoryBytes: projectedMemory }),
    ...(projectedContext === undefined ? {} : { contextLength: projectedContext }),
    installation: installation(model),
    ...(projectedResidency === undefined ? {} : { residency: projectedResidency }),
  })
}

export const modelsStatusJsonData = (result: ModelsStatusResult): ModelsStatusJsonData => {
  switch (result._tag) {
    case "Initializing": return { state: "initializing", view: result.view }
    case "List": return { state: "ready", view: "list", models: modelsForStatus(result.models).map(localModelJson) }
    case "Detail": return { state: "ready", view: "detail", model: localModelJson(result.model) }
  }
}
