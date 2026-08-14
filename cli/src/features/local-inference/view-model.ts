import { Option } from "effect"
import {
  installedLocalModels,
  localModelConfigurationId,
  localModelProviderModelId,
  formatLocalModelDisplayName,
  localModelBundleKey,
  localModelOptions,
  localModelSpeculativeMethodLabel,
  type LocalModelOption,
} from "@magnitudedev/client-common"
export {
  installedLocalModels,
  findLocalModelByConfigurationId,
  localModelConfigurationId,
  localModelProviderModelId,
  localModelBundleKey,
} from "@magnitudedev/client-common"
import {
  type LocalInferenceHardware,
  type LocalInferenceMemoryDomainId,
  type LocalModel,
  type ModelDownloadFailure,
  type LocalModelsState,
  type LocalModelRecommendationProgressStep,
  type ModelAssessmentId,
  type ModelServingConfigurationId,
  type ModelSlotsState,
  type ProviderModelId,
  type ReasoningEffort,
  servableModelBundlePackages,
} from "@magnitudedev/sdk"

export type LocalInferenceSelection = LocalModelOption

const modelPackages = (model: LocalModel) => servableModelBundlePackages(model.bundle)

export const localModelMaximumContextLength = (model: LocalModel): Option.Option<number> => {
  const known = modelPackages(model).flatMap(({ properties }) =>
    Option.match(properties.maximumContextLength, {
      onNone: () => [],
      onSome: (maximum) => [maximum],
    }))
  return known.length === 0 ? Option.none() : Option.some(Math.min(...known))
}

export const localModelDownloadBytes = (model: LocalModel): number =>
  model.downloadBytes

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

export const selectionConfigurationId = (
  selection: LocalInferenceSelection,
): Option.Option<ModelServingConfigurationId> => localModelConfigurationId(selection.model)

export const selectionProviderModelId = (
  selection: LocalInferenceSelection,
): Option.Option<ProviderModelId> => localModelProviderModelId(selection.model)

export const selectionReasoningEffort = (
  selection: LocalInferenceSelection,
): Option.Option<ReasoningEffort> => selection.model.servingState._tag === "Assessed"
  ? selection.model.servingState.capabilities.reasoning.defaultEffort
  : Option.none()

export const selectionAssessmentId = (
  selection: LocalInferenceSelection,
): Option.Option<ModelAssessmentId> => selection.model.servingState._tag === "Assessed"
  && selection.model.servingState.assessment._tag !== "Incompatible"
    ? Option.some(selection.model.servingState.assessment.assessmentId)
    : Option.none()

export const formatBytes = (bytes: number): string => {
  const gib = bytes / 1024 ** 3
  return gib >= 1 ? `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB` : `${(bytes / 1024 ** 2).toFixed(0)} MiB`
}

export const formatDownloadBytes = (bytes: number): string => {
  const gigabytes = bytes / 1_000_000_000
  return gigabytes >= 1
    ? `${gigabytes.toFixed(gigabytes >= 10 ? 1 : 2)} GB`
    : `${(bytes / 1_000_000).toFixed(0)} MB`
}

export const modelDownloadFailureMessage = (failure: ModelDownloadFailure): string => {
  switch (failure._tag) {
    case "Interrupted": return "The download was interrupted. Try again to continue."
    case "InsufficientDiskSpace":
      return `Not enough disk space. Free at least ${formatDownloadBytes(
        Math.max(0, failure.requiredBytes - failure.availableBytes),
      )} and try again.`
    case "SourceUnavailable": return "This model is not available from its source."
    case "NetworkUnavailable":
      return "Couldn’t reach the model source. Check your connection and try again."
    case "CorruptDownload":
      return "The downloaded file couldn’t be verified. Try the download again."
    case "LocalStorageFailure":
      return "Magnitude couldn’t write the model to disk. Check disk access and try again."
    case "Internal": return "Magnitude couldn’t complete the download. Try again."
  }
}

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
  if (model.servingState._tag !== "Assessed"
    || model.servingState.assessment._tag !== "Fits") {
    throw new Error("Performance requires a fitting assessed local model")
  }
  const { assessment } = model.servingState
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

const progressLabel = (
  step: LocalModelRecommendationProgressStep,
  completed: boolean,
): string => {
  if (step.id === "hardware") return completed ? "Detected hardware" : "Detecting hardware"
  if (step.id === "inventory") {
    if (!completed) return "Checking downloaded models"
    const count = Option.getOrElse(step.completedItems, () => 0)
    return `Found ${count} downloaded ${count === 1 ? "model" : "models"}`
  }
  if (step.id === "assessment") {
    if (!completed) return "Assessing models for this machine"
    const count = Option.getOrElse(step.completedItems, () => 0)
    return `Assessed ${count} models for this machine`
  }
  if (!completed) return "Preparing recommendations"
  const count = Option.getOrElse(step.completedItems, () => 0)
  return `Prepared ${count} recommendations`
}

const formatDurationMs = (durationMs: number): string => durationMs < 1_000
  ? `${(durationMs / 1_000).toFixed(1)}s`
  : durationMs < 60_000
    ? `${Math.round(durationMs / 1_000)}s`
    : `${Math.floor(durationMs / 60_000)}m ${Math.round(durationMs % 60_000 / 1_000)}s`

export interface LocalInferenceProgressLine {
  readonly id: LocalModelRecommendationProgressStep["id"]
  readonly state: "pending" | "running" | "completed" | "failed"
  readonly label: string
  readonly metadata: string
}

export const localInferenceProgressLines = (
  steps: readonly LocalModelRecommendationProgressStep[],
): readonly LocalInferenceProgressLine[] => steps.map((step) => {
  const completed = step.status._tag === "Completed"
  const label = progressLabel(step, completed)
  const showCount = step.id === "assessment" && step.status._tag === "Running"
  const count = showCount ? Option.match(step.totalItems, {
    onNone: () => "",
    onSome: (total) => Option.match(step.completedItems, {
      onNone: () => ` · ${total}`,
      onSome: (value) => ` · ${value}/${total}`,
    }),
  }) : ""
  if (step.status._tag === "Pending") {
    return { id: step.id, state: "pending", label, metadata: "" }
  }
  if (step.status._tag === "Running") {
    const estimate = Option.match(step.estimatedRemainingMs, {
      onNone: () => "",
      onSome: (remainingMs) => ` · about ${formatDurationMs(remainingMs)} left`,
    })
    return {
      id: step.id,
      state: "running",
      label,
      metadata: `${count}${estimate}`,
    }
  }
  if (step.status._tag === "Failed") {
    return {
      id: step.id,
      state: "failed",
      label: `${label} failed`,
      metadata: ` · ${step.status.failure.message}`,
    }
  }
  return {
    id: step.id,
    state: "completed",
    label,
    metadata: step.id === "assessment" && !step.status.cached
      ? ` · ${formatDurationMs(step.status.durationMs)}`
      : "",
  }
})

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

const compactBytes = (bytes: number): string => formatBytes(bytes).replace(/\.0+(?= )/, "")

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
      ? `${compactBytes(hardware.totalSystemMemoryBytes)} unified`
      : `${compactBytes(hardware.totalSystemMemoryBytes)} RAM`,
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
        details: [`${compactBytes(domain.totalBytes)} VRAM`, ...backends],
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
        `${formatBytes(hardware.totalSystemMemoryBytes)} ${unified.length > 0 ? "unified" : "system"} memory${unifiedBackends.length > 0 ? ` · ${unifiedBackends.join(" + ")} GPU acceleration` : ""}`,
      ],
    },
    accelerators: discrete.map((domain) => {
      const names = namesFor(domain.memoryDomainId)
      const backends = backendsFor(domain.memoryDomainId)
      return {
        name: names.join(" + ") || `${backends[0] ?? "Local"} GPU`,
        details: `${formatBytes(domain.totalBytes)} VRAM · ${backends.join(" + ") || "GPU"} acceleration`,
      }
    }),
  }
}

export const selectionTitle = ({ model }: LocalInferenceSelection): string =>
  formatLocalModelDisplayName(model)

export const selectionMetadata = ({ model }: LocalInferenceSelection): string => {
  const configuration = model.servingState._tag === "Resolving"
    ? Option.none()
    : model.servingState._tag === "Failed"
      ? model.servingState.configuration
      : Option.some(model.servingState.configuration)
  const contextLength = Option.match(configuration, {
    onNone: () => localModelMaximumContextLength(model),
    onSome: ({ profile }) => Option.some(profile.contextLength),
  })
  const speculativeMethod = Option.getOrNull(localModelSpeculativeMethodLabel(model))
  return [
    formatDownloadBytes(model.downloadBytes),
    Option.map(contextLength, (context) => `${formatContext(context)} ctx`).pipe(Option.getOrNull),
    speculativeMethod,
  ].filter((value): value is string => value !== null).join(" · ")
}

export const selectionCapacityWarning = ({ model }: LocalInferenceSelection): string | null =>
  model.servingState._tag === "Assessed"
    && model.servingState.availabilityState._tag === "Unavailable"
    && model.servingState.availabilityState.failure.code === "insufficient_resources"
    ? model.servingState.availabilityState.failure.message
    : null
