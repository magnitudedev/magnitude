import { Option } from "effect"
import {
  installedLocalModels,
  localModelProviderModelId,
  localModelStorageBytes,
  localModelServingState,
  localModelServingProfile,
  formatLocalModelDisplayName,
  localModelOptions,
  localModelSpeculativeMethodLabel,
  formatStorageSize,
  formatMemorySize,
  type LocalModelOption,
} from "@magnitudedev/client-common"
export {
  installedLocalModels,
  findLocalModelById,
  localModelProviderModelId,
  modelDownloadFailureMessage,
} from "@magnitudedev/client-common"
import {
  type LocalInferenceHardware,
  type LocalInferenceMemoryDomainId,
  type LocalModel,
  type LocalModelsState,
  type ModelAssessmentId,
  type ModelId,
  type ModelSlotsState,
  type ProviderModelId,
  type ReasoningEffort,
} from "@magnitudedev/sdk"

export type LocalInferenceSelection = LocalModelOption

export const localModelMaximumContextLength = (model: LocalModel): Option.Option<number> => {
  return Option.flatMap(localModelServingState(model), (serving) => {
    if (serving._tag === "Assessed") {
      return Option.orElse(serving.metadata.maximumContextLength, () => Option.some(serving.assessment.profile.contextLength))
    }
    if (serving._tag === "Assessing") return Option.some(serving.profile.contextLength)
    return Option.map(localModelServingProfile(model), ({ contextLength }) => contextLength)
  })
}

export const buildLocalInferenceSelections = (
  models: LocalModelsState,
  slots: ModelSlotsState,
): readonly LocalInferenceSelection[] => localModelOptions(models, slots)

export const selectedInferenceIndex = (
  selections: readonly LocalInferenceSelection[],
  selectedId: Option.Option<string>,
): number => {
  const index = Option.match(selectedId, {
    onNone: () => -1,
    onSome: (id) => selections.findIndex((selection) => selection.id === id),
  })
  return index >= 0 ? index : 0
}

export const selectionModelId = (
  selection: LocalInferenceSelection,
): ModelId => selection.model.modelId

export const selectionProviderModelId = (
  selection: LocalInferenceSelection,
): Option.Option<ProviderModelId> => localModelProviderModelId(selection.model)

export const selectionReasoningEffort = (
  selection: LocalInferenceSelection,
): Option.Option<ReasoningEffort> => Option.flatMap(localModelServingState(selection.model), (serving) =>
  serving._tag === "Assessed" ? serving.capabilities.reasoning.defaultEffort : Option.none())

export const selectionAssessmentId = (
  selection: LocalInferenceSelection,
): Option.Option<ModelAssessmentId> => Option.flatMap(localModelServingState(selection.model), (serving) =>
  serving._tag === "Assessed" && serving.assessment._tag !== "Incompatible"
    ? Option.some(serving.assessment.assessmentId)
    : Option.none())

export const formatContext = (tokens: number): string => tokens < 1_000
  ? String(tokens)
  : tokens % 1_024 === 0
    ? `${tokens / 1_024}K`
    : `${Math.round(tokens / 1_000)}K`

export const performanceRange = (
  model: LocalModel,
): {
  readonly lowerContext: number
  readonly upperContext: number
  readonly lowerTokensPerSecond: number
  readonly upperTokensPerSecond: number
} => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed"
    || serving.assessment._tag !== "Fits") {
    throw new Error("Performance requires a fitting assessed local model")
  }
  const { assessment } = serving
  const lowerContext = Math.min(25_000, assessment.profile.contextLength)
  const upperContext = Math.min(75_000, assessment.profile.contextLength)
  const lowerSample = assessment.performance.find(({ contextTokens }) =>
    contextTokens === lowerContext)!
  const upperSample = assessment.performance.find(({ contextTokens }) =>
    contextTokens === upperContext)!
  return {
    lowerContext,
    upperContext,
    lowerTokensPerSecond: Math.min(
      lowerSample.estimatedTokensPerSecond,
      upperSample.estimatedTokensPerSecond,
    ),
    upperTokensPerSecond: Math.max(
      lowerSample.estimatedTokensPerSecond,
      upperSample.estimatedTokensPerSecond,
    ),
  }
}

export const performanceRangeSpeedLabel = (
  model: LocalModel,
  unit = "tok/s",
): string => {
  const range = performanceRange(model)
  return Math.round(range.lowerTokensPerSecond) === Math.round(range.upperTokensPerSecond)
    ? `~${Math.round(range.lowerTokensPerSecond)} ${unit}`
    : `~${Math.round(range.lowerTokensPerSecond)}–${Math.round(range.upperTokensPerSecond)} ${unit}`
}


export interface LocalHardwarePresentation {
  readonly system: { readonly name: string; readonly details: readonly string[] }
  readonly accelerators: readonly { readonly name: string; readonly details: string }[]
}

export interface LocalHardwareSummaryRow {
  readonly name: string
  readonly details: readonly string[]
}

const unique = (values: readonly string[]): string[] =>
  [...new Set(values.map((value) => value.trim()).filter(Boolean))]

const localHardwareTopology = (hardware: LocalInferenceHardware) => {
  const unified = hardware.memoryDomains.filter((domain) =>
    domain.kind === "UnifiedMemory" && domain.sharesSystemMemory)
  const discrete = hardware.memoryDomains.filter((domain) => domain.kind === "PhysicalDevice")
  const backendsFor = (memoryDomainId: LocalInferenceMemoryDomainId) => unique(hardware.accelerators
    .filter((accelerator) => accelerator.memoryDomainId === memoryDomainId)
    .map((accelerator) => accelerator.backend))
  const namesFor = (memoryDomainId: LocalInferenceMemoryDomainId) => unique(hardware.accelerators
    .filter((accelerator) => accelerator.memoryDomainId === memoryDomainId)
    .map((accelerator) => accelerator.name))
  const unifiedBackends = unique(unified.flatMap((domain) => backendsFor(domain.memoryDomainId)))
  const unifiedAcceleratorNames = unique(unified.flatMap((domain) =>
    namesFor(domain.memoryDomainId)))
  const isAppleSilicon = hardware.platform === "MacOS" && hardware.architecture === "Arm64"
  const processorName = Option.getOrElse(hardware.processor, () =>
    isAppleSilicon ? "Apple Silicon" : "CPU")
  const productName = Option.getOrElse(hardware.productName, () => "")
  const systemName = isAppleSilicon
    ? processorName
    : unique([productName, unifiedAcceleratorNames.join(" + ")]).join(" · ") || processorName
  return { unified, discrete, backendsFor, namesFor, unifiedBackends, systemName }
}

export const describeLocalHardwareSummary = (
  hardware: LocalInferenceHardware,
): readonly LocalHardwareSummaryRow[] => {
  const {
    unified,
    discrete,
    backendsFor,
    namesFor,
    unifiedBackends,
    systemName,
  } = localHardwareTopology(hardware)
  const platform = hardware.platform === "MacOS" ? "macOS" : hardware.platform
  const architecture = hardware.architecture === "Arm64" ? "ARM64" : "x86-64"
  const systemDetails = [
    `${platform} ${architecture}`,
    `${hardware.logicalCores} ${hardware.logicalCores === 1 ? "core" : "cores"}`,
    unified.length > 0
      ? `${formatMemorySize(hardware.totalSystemMemoryBytes)} unified`
      : `${formatMemorySize(hardware.totalSystemMemoryBytes)} RAM`,
    ...unifiedBackends,
    ...(unified.length === 0 && discrete.length === 0 ? ["CPU inference"] : []),
  ]
  return [
    { name: systemName, details: systemDetails },
    ...discrete.map((domain): LocalHardwareSummaryRow => {
      const names = namesFor(domain.memoryDomainId)
      const backends = backendsFor(domain.memoryDomainId)
      return {
        name: names.join(" + ") || `${backends[0] ?? "Local"} GPU`,
        details: [`${formatMemorySize(domain.totalBytes)} VRAM`, ...backends],
      }
    }),
  ]
}

export const describeLocalHardware = (
  hardware: LocalInferenceHardware,
): LocalHardwarePresentation => {
  const {
    unified,
    discrete,
    backendsFor,
    namesFor,
    unifiedBackends,
    systemName,
  } = localHardwareTopology(hardware)
  return {
    system: {
      name: systemName,
      details: [
        `${hardware.platform === "MacOS" ? "macOS" : hardware.platform} · ${hardware.architecture === "Arm64" ? "ARM64" : "x86-64"} · ${hardware.logicalCores} logical CPU core${hardware.logicalCores === 1 ? "" : "s"}`,
        `${formatMemorySize(hardware.totalSystemMemoryBytes)} ${unified.length > 0 ? "unified" : "system"} memory${unifiedBackends.length > 0 ? ` · ${unifiedBackends.join(" + ")} GPU acceleration` : ""}`,
      ],
    },
    accelerators: discrete.map((domain) => {
      const names = namesFor(domain.memoryDomainId)
      const backends = backendsFor(domain.memoryDomainId)
      return {
        name: names.join(" + ") || `${backends[0] ?? "Local"} GPU`,
        details: `${formatMemorySize(domain.totalBytes)} VRAM · ${backends.join(" + ") || "GPU"} acceleration`,
      }
    }),
  }
}

export const selectionTitle = ({ model }: LocalInferenceSelection): string =>
  formatLocalModelDisplayName(model)

export const selectionContextLabel = ({ model }: LocalInferenceSelection): Option.Option<string> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving === undefined) return localModelMaximumContextLength(model).pipe(Option.map(formatContext))
  const context = serving._tag === "Failed"
    ? Option.map(localModelServingProfile(model), ({ contextLength }) => contextLength)
    : Option.some(serving._tag === "Assessed"
        ? serving.assessment.profile.contextLength
        : serving.profile.contextLength)
  return Option.orElse(context, () => localModelMaximumContextLength(model)).pipe(Option.map(formatContext))
}

export const selectionMetadata = (selection: LocalInferenceSelection): string => {
  const { model } = selection
  const speculativeMethod = Option.getOrNull(localModelSpeculativeMethodLabel(model))
  return [
    Option.map(localModelStorageBytes(model), formatStorageSize).pipe(Option.getOrNull),
    Option.map(selectionContextLabel(selection), (context) => `${context} ctx`).pipe(Option.getOrNull),
    speculativeMethod,
  ].filter((value): value is string => value !== null).join(" · ")
}

export const selectionCapacityWarning = ({ model }: LocalInferenceSelection): string | null =>
  Option.match(localModelServingState(model), {
    onNone: () => null,
    onSome: (serving) => serving._tag === "Assessed" && serving.assessment._tag === "DoesNotFit"
      ? "This model does not fit available hardware."
      : null,
  })
