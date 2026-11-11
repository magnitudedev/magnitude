import {
  catalogModelReplacement,
  describeCatalogDeprecation,
  fitPlacementPhrase,
  formatQuantLabel,
  ladderBitWidths,
  formatMemorySize,
  formatStorageSize,
  formatModelLoadPercentage,
  formatModelLoadStage,
  formatSpeculativeMethod,
  isMeasuredModelLoadStage,
  MODEL_STOPPED_FOR_MEMORY_MESSAGE,
  describeModelLoadFailure,
  modelStoppedForMemory,
  localModelResidency,
  localModelContextLimit,
  localModelMemoryParts,
  localModelServingProfile,
  localModelServingState,
  localModelUnservedContext,
  type ServingMemory,
} from "@magnitudedev/client-common"
import {
  comparisonTokensPerSecond,
  findLocalModelById,
  installedAcquisition,
  type CatalogLocalModel,
  type LocalModel,
  type LocalModelAssessment,
  type LocalModelFitsAssessment,
  type ModelCatalogState,
  type ModelId,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { addressableModels } from "./inference-client"
import { formatContext } from "./inference-format"
import { ModelCommandError } from "./model-references"
import { renderFields, renderTable } from "./output"

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

/** Why this computer serves a model at no context: what it lacks at the smallest one. */
export const doesNotFitLabel = (model: LocalModel): Option.Option<string> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  return serving?._tag === "Assessed" && serving.assessment._tag === "DoesNotFit"
    ? Option.some(`Doesn't fit - needs ${formatMemorySize(serving.assessment.deficitBytes, { rounding: "up" })} more, even at ${formatContext(serving.assessment.shortfall.contextLength)}`)
    : Option.none()
}

/** The context a model is served at, with its own when this computer serves it at less. */
export const modelContext = (model: LocalModel): string => Option.isSome(doesNotFitLabel(model)) ? "—" : Option.match(localModelServingProfile(model), {
  onNone: () => "—",
  onSome: ({ contextLength }) => Option.match(localModelContextLimit(model), {
    onNone: () => formatContext(contextLength),
    onSome: (limit) => limit.contextLength > contextLength
      ? `${formatContext(contextLength)} of ${formatContext(limit.contextLength)}`
      : formatContext(contextLength),
  }),
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
  if (bytes === undefined) return "—"
  const serving = Option.getOrUndefined(localModelServingState(model))
  return serving?._tag === "Assessed" && serving.assessment._tag === "DoesNotFit"
    ? `${formatMemorySize(bytes)} at ${formatContext(serving.assessment.shortfall.contextLength)}`
    : formatMemorySize(bytes)
}

/** What a fitting model's memory holds, as the desktop shows it: "Weights 36 GB · 1M context 7.1 GB · Overhead 2.3 GB". */
export const memoryBreakdownFields = (model: LocalModel): readonly (readonly [string, string])[] => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed" || serving.assessment._tag !== "Fits") return []
  const parts = localModelMemoryParts(serving.assessment).map(({ label, bytes }) => `${label} ${formatMemorySize(bytes)}`).join(" · ")
  return [["Breakdown", Option.match(localModelUnservedContext(serving.assessment), {
    onNone: () => parts,
    onSome: (unserved) => `${parts} (${unserved})`,
  })]]
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
      ["Switch", `magnitude models pull ${support.replacement}`],
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

/** The Hugging Face repository a model's files come from: a catalog quant's, an added model's, or a found model's `hf:` ID. */
export const modelRepository = (model: LocalModel): Option.Option<string> => {
  if (model._tag === "Catalog") return Option.some(model.catalogData.quant.repository)
  if (model.origin._tag === "Added") return Option.some(model.origin.repository)
  const [form, owner, repo] = model.modelId.split(/[:/]/)
  return form === "hf" && owner !== undefined && repo !== undefined ? Option.some(`${owner}/${repo}`) : Option.none()
}

const quantName = (model: LocalModel): string =>
  model._tag === "Catalog" ? model.catalogData.quant.name : model.presentation.variantLabel

/** A model's quant: "4-bit unsloth UD-Q4_K_XL", with the bit width a ladder gave it when there is one. */
export const modelQuant = (model: LocalModel, bits?: Option.Option<number>): string =>
  formatQuantLabel(quantName(model), modelRepository(model), bits)

/** Where a model's files are: Magnitude's own downloads, or the other app or folder Magnitude found them in. */
export const modelSource = (model: LocalModel): string => {
  if (model._tag === "Catalog") {
    return installedAcquisition(model.acquisitionState)?.installation.ownership === "External" ? "Another app" : "Magnitude"
  }
  if (model.origin._tag === "Added") return "Magnitude"
  switch (model.origin.source._tag) {
    case "HuggingFaceCache": return "HF cache"
    case "LmStudio": return "LM Studio"
    case "LlamaCpp": return "llama.cpp"
    case "Ollama": return "Ollama"
    case "Folder": return "folder"
  }
}

/** A model's assessment on this computer, once it is assessed. */
export const modelAssessment = (model: LocalModel): Option.Option<LocalModelAssessment> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  return serving?._tag === "Assessed" ? Option.some(serving.assessment) : Option.none()
}

const fits = (assessment: Option.Option<LocalModelAssessment>): Option.Option<LocalModelFitsAssessment> =>
  Option.filter(assessment, (value): value is LocalModelFitsAssessment => value._tag === "Fits")

const tokensPerSecond = (value: number): string => `~${Math.round(value)} tok/s`

/** The estimated decode speed of an assessed quant at the depth models are compared at ("~28 tok/s"). */
export const assessmentSpeed = (assessment: Option.Option<LocalModelAssessment>): string => Option.match(assessment, {
  onNone: () => "—",
  onSome: (value) => {
    switch (value._tag) {
      case "DoesNotFit": return "Doesn't fit"
      case "Unsupported": return "Can't run"
      case "Fits": {
        const { performance } = value.residency
        if (performance._tag === "Pending") return "Measuring…"
        return Option.match(comparisonTokensPerSecond(performance, value.served.contextLength), {
          onNone: () => "—",
          onSome: tokensPerSecond,
        })
      }
    }
  },
})

/** Whether an assessed quant's decode speed is known: it fits and this machine's speed is measured. */
export const assessmentSpeedKnown = (assessment: Option.Option<LocalModelAssessment>): boolean => Option.exists(fits(assessment), (value) =>
  value.residency.performance._tag !== "Pending" && Option.isSome(comparisonTokensPerSecond(value.residency.performance, value.served.contextLength)))

/** The context an assessed quant is served at, with the model's own when this computer serves less ("32K of 128K"). */
export const assessmentContext = (assessment: Option.Option<LocalModelAssessment>): string => Option.match(fits(assessment), {
  onNone: () => "—",
  onSome: ({ served, profile }) => profile.contextLength > served.contextLength
    ? `${formatContext(served.contextLength)} of ${formatContext(profile.contextLength)}`
    : formatContext(served.contextLength),
})

/** How an assessed quant runs on this machine: "Fits in memory", "Streams from SSD". */
const servingMemoryLabel: Record<ServingMemory, string> = { Unified: "Unified memory", Device: "GPU", System: "RAM" }

export const assessmentPlacement = (assessment: Option.Option<LocalModelAssessment>, memory: ServingMemory): string =>
  Option.match(fits(assessment), { onNone: () => "—", onSome: (value) => fitPlacementPhrase(value, memory) })

/** Where a fitting quant's bytes are, by tier, and its decode speed at each assessed context depth. */
export const fitDetailSections = (assessment: Option.Option<LocalModelAssessment>, memory: ServingMemory): string[] =>
  Option.match(fits(assessment), {
    onNone: () => [],
    onSome: (value) => [
      "",
      `Placement - ${fitPlacementPhrase(value, memory)}`,
      renderFields([
        [servingMemoryLabel[memory], formatMemorySize(value.placement.deviceBytes)],
        ...(memory !== "Device" || value.placement.hostBytes === 0 ? [] : [["RAM", formatMemorySize(value.placement.hostBytes)] as const]),
        ...(value.placement.diskBytes === 0 ? [] : [["Streamed from SSD", formatMemorySize(value.placement.diskBytes)] as const]),
        ...(value.placement.lookupDiskBytes === 0 ? [] : [["Stays on SSD", formatMemorySize(value.placement.lookupDiskBytes)] as const]),
      ]),
      "",
      "Estimated decode speed",
      value.residency.performance._tag === "Estimated"
        ? renderFields(value.residency.performance.samples.map(({ contextTokens, estimatedTokensPerSecond }) => [
            `${formatContext(contextTokens)} context${contextTokens === value.served.contextLength ? " (served)" : ""}`,
            tokensPerSecond(estimatedTokensPerSecond),
          ] as const))
        : renderFields([["Status", value.residency.performance._tag === "Pending" ? "Measuring this machine…" : value.residency.performance.failure.message]]),
    ],
  })

/** A model's quants as this computer would run each one, with the one `chosen` names marked. */
export const renderQuantLadder = (
  rungs: readonly LocalModel[],
  chosen: Option.Option<string>,
  memory: ServingMemory,
): string => {
  const widths = ladderBitWidths(rungs.map(quantName))
  return renderTable(rungs, [
    { heading: "QUANT", value: (model) => `${modelQuant(model, widths[rungs.indexOf(model)])}${Option.contains(chosen, model.modelId as string) ? " *" : ""}` },
    { heading: "SIZE", value: (model) => formatStorageSize(model.storageBytes) },
    { heading: "SPEED", value: (model) => assessmentSpeed(modelAssessment(model)) },
    { heading: "CONTEXT", value: (model) => assessmentContext(modelAssessment(model)) },
    { heading: "PLACEMENT", value: (model) => assessmentPlacement(modelAssessment(model), memory) },
    { heading: "MODEL ID", value: ({ modelId }) => modelId },
  ])
}
