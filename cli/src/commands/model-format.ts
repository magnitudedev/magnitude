import {
  catalogModelReplacement,
  describeCatalogDeprecation,
  formatMemorySize,
  formatModelLoadPercentage,
  formatModelLoadStage,
  formatSpeculativeMethod,
  isMeasuredModelLoadStage,
  MODEL_STOPPED_FOR_MEMORY_MESSAGE,
  describeModelLoadFailure,
  modelStoppedForMemory,
  localModelResidency,
  localModelServingProfile,
  localModelServingState,
} from "@magnitudedev/client-common"
import { findLocalModelById, type CatalogLocalModel, type LocalModel, type ModelCatalogState, type ModelId } from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { addressableModels } from "./inference-client"
import { formatContext } from "./inference-format"
import { ModelCommandError } from "./model-references"

export const optional = <A, I>(schema: Schema.Schema<A, I, never>) =>
  Schema.optionalWith(schema, { as: "Option", exact: true })

export const plural = (count: number, singular: string, pluralForm = `${singular}s`) =>
  `${count} ${count === 1 ? singular : pluralForm}`

/** The models a command can address by ID, hidden ones included. */
export const requireLocalModels = (catalog: ModelCatalogState): Effect.Effect<readonly LocalModel[], ModelCommandError> =>
  catalog._tag === "Initializing"
    ? Effect.fail(new ModelCommandError({ message: "Local models are initializing. Try again shortly." }))
    : Effect.succeed(addressableModels(catalog))

export const findModel = <M extends LocalModel>(
  models: readonly LocalModel[],
  modelId: string,
  kind = "model",
  accept: (model: LocalModel) => model is M = (model): model is M => true,
): Effect.Effect<M, ModelCommandError> => {
  const model = Option.getOrUndefined(findLocalModelById(models, modelId as ModelId))
  return model === undefined || !accept(model)
    ? Effect.fail(new ModelCommandError({ message: `Unknown ${kind}: ${modelId}` }))
    : Effect.succeed(model)
}

export const isCatalogModel = (model: LocalModel): model is CatalogLocalModel => model._tag === "Catalog"

export const residencyState = (model: LocalModel) => Option.getOrUndefined(localModelResidency(model))

export const modelContext = (model: LocalModel): string => Option.match(localModelServingProfile(model), {
  onNone: () => "—",
  onSome: ({ contextLength }) => formatContext(contextLength),
})

export const modelMemoryBytes = (model: LocalModel): number | undefined => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed") return undefined
  return serving.assessment._tag === "Fits"
    ? serving.assessment.memory.totalRequiredBytes
    : serving.assessment._tag === "DoesNotFit" ? serving.assessment.totalRequiredBytes : undefined
}

export const modelMemory = (model: LocalModel): string => {
  const bytes = modelMemoryBytes(model)
  return bytes === undefined ? "—" : formatMemorySize(bytes)
}

export const supportFields = (
  model: LocalModel,
  models: readonly LocalModel[],
): readonly (readonly [string, string])[] => {
  if (model._tag !== "Catalog") return []
  const support = model.catalogData.support
  switch (support._tag) {
    case "Supported": return []
    case "Disabled": return [["Support", `Disabled - ${support.reason}`]]
    case "Deprecated": return [
      ["Support", describeCatalogDeprecation(support, catalogModelReplacement(models.filter(isCatalogModel), support))],
      ["Switch", `magnitude catalog pull ${support.replacement}`],
    ]
  }
}

export const accelerationLabel = (model: LocalModel): string => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  return serving?._tag === "Assessed"
    ? Option.match(serving.speculativeMethod, { onNone: () => "None", onSome: formatSpeculativeMethod })
    : "None"
}

export const capabilityLabels = (model: LocalModel): string[] => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed") return []
  const capabilities = serving.capabilities
  return [
    ...(capabilities.vision ? ["Vision"] : []),
    ...(capabilities.tools ? ["tools"] : []),
    ...(capabilities.structuredOutput ? ["structured output"] : []),
    ...(capabilities.reasoning.supported ? ["reasoning"] : []),
  ]
}

export const UNLOADED = "Unloaded"

export const residencyLabel = (model: LocalModel): string => {
  const residency = residencyState(model)
  if (residency === undefined) return UNLOADED
  switch (residency._tag) {
    case "Unloaded": return UNLOADED
    case "Stopped": return modelStoppedForMemory(residency) ? MODEL_STOPPED_FOR_MEMORY_MESSAGE : UNLOADED
    case "Requested": return "Loading"
    case "Loading": return isMeasuredModelLoadStage(residency.stage)
      ? `${formatModelLoadStage(residency.stage)} ${formatModelLoadPercentage(residency.fraction)}`
      : formatModelLoadStage(residency.stage)
    case "Failed": return `Failed - ${describeModelLoadFailure(residency.failure)}`
    default: return residency._tag
  }
}

export const percent = (completed: number, total: number): string =>
  `${total === 0 ? 0 : Math.round((completed / total) * 100)}%`
