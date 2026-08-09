import { Option } from "effect"
import {
  ProviderModelCatalogLifecycle,
  ProviderIdSchema,
  type LocalInferenceHardware,
  type LocalInferenceMemoryDomainId,
  type LocalModel,
  type LocalModelCatalogCandidate,
  type LocalModelsState,
  type LocalModelRecommendation,
  type ModelServingConfigurationId,
  type LocalModelRecommendationProgressStep,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type ProviderModelId,
  type ReasoningEffort,
} from "@magnitudedev/sdk"

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")

type LocalInferenceSelectionBase = {
  readonly id: string
  readonly model: LocalModel
  readonly contextLength: number
  readonly providerModelId: Option.Option<ProviderModelId>
  readonly reasoningEffort: Option.Option<ReasoningEffort>
}

export interface InstalledLocalModelChoice {
  readonly id: string
  readonly model: LocalModel
  readonly configurationId: ModelServingConfigurationId
  readonly candidate: Option.Option<LocalModelCatalogCandidate>
  readonly providerModel: Option.Option<ProviderModelCatalogEntry>
  readonly running: boolean
  readonly contextLength: number
  readonly reasoningEffort: Option.Option<ReasoningEffort>
}

export type LocalInferenceSelection = LocalInferenceSelectionBase & (
  | {
      readonly kind: "running" | "stored"
      readonly configurationId: ModelServingConfigurationId
      readonly recommendation: { readonly _tag: "None" }
    }
  | {
      readonly kind: "recommendation"
      readonly recommendation:
        | { readonly _tag: "None" }
        | { readonly _tag: "Recommended"; readonly value: LocalModelRecommendation }
    }
)

const selectionKindOrder: Record<LocalInferenceSelection["kind"], number> = {
  running: 0,
  stored: 1,
  recommendation: 2,
}

const recommendationIntentOrder = {
  balanced: 0,
  best_quality: 1,
  fastest: 2,
  lightweight: 3,
} as const

const compareSelections = (
  left: LocalInferenceSelection,
  right: LocalInferenceSelection,
): number => selectionKindOrder[left.kind] - selectionKindOrder[right.kind]
  || (left.kind === "recommendation" && right.kind === "recommendation"
    ? (left.recommendation._tag === "Recommended"
      ? recommendationIntentOrder[left.recommendation.value.intent]
      : 4) - (right.recommendation._tag === "Recommended"
      ? recommendationIntentOrder[right.recommendation.value.intent]
      : 4)
    : 0)
  || left.model.displayName.localeCompare(right.model.displayName)

const providerCatalogModels = (
  catalog: ProviderModelCatalogState,
): readonly ProviderModelCatalogEntry[] => ProviderModelCatalogLifecycle.match(catalog, {
  Loading: () => [],
  Ready: ({ models }) => models,
  Refreshing: ({ models }) => models,
  Degraded: ({ models }) => models,
  Unavailable: () => [],
})

export const buildInstalledLocalModelChoices = (
  models: LocalModelsState,
  catalog: ProviderModelCatalogState,
  slots: ModelSlotsState,
): readonly InstalledLocalModelChoice[] => {
  const selectedProviderModelIds = new Set([slots.slots.primary, slots.slots.secondary].flatMap((slot) =>
    slot._tag === "ConfiguredLocal" ? [slot.selection.providerModelId] : []))
  const running = new Set([slots.slots.primary, slots.slots.secondary].flatMap((slot) =>
    slot._tag === "ConfiguredLocal"
      && Option.exists(slot.instance, (instance) => instance.lifecycle._tag === "Ready")
      ? [slot.selection.providerModelId]
      : []))
  const catalogModels = providerCatalogModels(catalog)
  const localProviderModels = new Map(catalogModels
    .filter(({ providerId }) => providerId === LOCAL_PROVIDER_ID)
    .map((model) => [model.providerModelId, model]))
  const installedCandidates = models.recommendations._tag === "Ready"
    ? models.recommendations.catalog.reduce((selected, candidate) => {
        if (candidate.download._tag !== "Downloaded") return selected
        const current = selected.get(candidate.targetId)
        if (current === undefined
          || (current.availability._tag !== "Available"
            && candidate.availability._tag === "Available")) {
          selected.set(candidate.targetId, candidate)
        }
        return selected
      }, new Map<string, LocalModelCatalogCandidate>())
    : new Map<string, LocalModelCatalogCandidate>()

  return models.models.flatMap((model): readonly InstalledLocalModelChoice[] => {
    if (model.download._tag !== "Downloaded") return []
    const candidate = installedCandidates.get(model.targetId)
    const offerings = model.offerings.flatMap((offering) => {
      const providerModel = localProviderModels.get(offering.providerModelId)
      return providerModel === undefined ? [] : [{ offering, providerModel }]
    })
    const selected = offerings.find(({ offering }) => selectedProviderModelIds.has(offering.providerModelId))
      ?? offerings.find(({ offering }) => offering.configurationId === candidate?.configurationId)
      ?? (candidate === undefined
        ? offerings.find(({ providerModel }) => providerModel.availability._tag === "Available")
          ?? offerings[0]
        : undefined)
    if (selected === undefined && candidate === undefined) return []
    const configurationId = selected?.offering.configurationId ?? candidate!.configurationId
    const providerModel = Option.fromNullable(selected?.providerModel)
    const catalogCandidate = Option.fromNullable(candidate)
    return [{
      id: `installed:${model.targetId}`,
      model,
      configurationId,
      candidate: catalogCandidate,
      providerModel,
      running: selected !== undefined && running.has(selected.offering.providerModelId),
      contextLength: selected?.providerModel.contextWindow
        ?? candidate?.profile.contextLength
        ?? model.maximumContextLength,
      reasoningEffort: selected?.providerModel.capabilities.reasoning.defaultEffort
        ?? candidate?.capabilities.reasoning.defaultEffort
        ?? Option.none(),
    }]
  })
}

export const buildLocalInferenceSelections = (
  models: LocalModelsState,
  catalog: ProviderModelCatalogState,
  slots: ModelSlotsState,
): readonly LocalInferenceSelection[] => {
  const installed = buildInstalledLocalModelChoices(models, catalog, slots)
    .flatMap((choice): readonly LocalInferenceSelection[] => {
      const availableProviderModel = Option.filter(
        choice.providerModel,
        ({ availability }) => availability._tag === "Available",
      )
      const availableCandidate = Option.filter(
        choice.candidate,
        ({ availability }) => availability._tag === "Available",
      )
      if (Option.isNone(availableProviderModel) && Option.isNone(availableCandidate)) return []
      return [{
        id: choice.id,
        kind: choice.running ? "running" : "stored",
        model: choice.model,
        configurationId: choice.configurationId,
        recommendation: { _tag: "None" },
        providerModelId: Option.map(availableProviderModel, ({ providerModelId }) => providerModelId),
        contextLength: choice.contextLength,
        reasoningEffort: choice.reasoningEffort,
      }]
    })
  const recommendations = models.recommendations._tag === "Ready"
    ? models.recommendations.entries.flatMap((recommendation): readonly LocalInferenceSelection[] => {
        const model = models.models.find(({ targetId }) =>
          targetId === recommendation.candidate.targetId)
        if (!model || model.download._tag === "Downloaded") return []
        return [{
          id: `recommendation:${recommendation.id}`,
          kind: "recommendation",
          model,
          recommendation: { _tag: "Recommended", value: recommendation },
          contextLength: recommendation.candidate.profile.contextLength,
          providerModelId: Option.none(),
          reasoningEffort: recommendation.candidate.capabilities.reasoning.defaultEffort,
        }]
      })
    : []
  const representedModelIds = new Set(recommendations.map(({ model }) => model.targetId))
  const transientDownloads = models.models
    .filter((model) =>
      (model.download._tag === "Downloading" || model.download._tag === "Failed")
      && !representedModelIds.has(model.targetId))
    .map((model): LocalInferenceSelection => ({
      id: `download:${model.targetId}`,
      kind: "recommendation",
      model,
      recommendation: { _tag: "None" },
      contextLength: model.maximumContextLength,
      providerModelId: Option.fromNullable(model.offerings[0]?.providerModelId),
      reasoningEffort: Option.none(),
    }))
  return [...installed, ...recommendations, ...transientDownloads]
    .sort(compareSelections)
}

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

export const formatContext = (tokens: number): string => tokens < 1_000
  ? String(tokens)
  : tokens % 1_024 === 0
    ? `${tokens / 1_024}K`
    : `${Math.round(tokens / 1_000)}K`

export const performanceRange = (
  candidate: LocalModelCatalogCandidate,
): {
  readonly lowerContext: number
  readonly upperContext: number
  readonly lowerTokensPerSecond: number
  readonly upperTokensPerSecond: number
} => {
  const lowerContext = Math.min(25_000, candidate.profile.contextLength)
  const upperContext = Math.min(75_000, candidate.profile.contextLength)
  const lowerSample = candidate.performance.find(({ contextTokens }) =>
    contextTokens === lowerContext)!
  const upperSample = candidate.performance.find(({ contextTokens }) =>
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
  candidate: LocalModelCatalogCandidate,
  unit = "tok/s",
): string => {
  const range = performanceRange(candidate)
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

export const selectionTitle = ({ model }: LocalInferenceSelection): string => model.displayName

export const selectionMetadata = ({ model, contextLength }: LocalInferenceSelection): string =>
  `${model.quantization} · ${formatDownloadBytes(model.downloadBytes)} · ${formatContext(
    contextLength,
  )} ctx`

export const selectionCapacityWarning = ({ recommendation }: LocalInferenceSelection): string | null =>
  recommendation._tag === "Recommended"
    && recommendation.value.candidate.availability._tag === "Unavailable"
    && recommendation.value.candidate.availability.failure.code === "insufficient_resources"
    ? recommendation.value.candidate.availability.failure.message
    : null
