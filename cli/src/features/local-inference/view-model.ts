import { Option } from "effect"
import {
  type LocalInferenceHardware,
  type LocalInferenceMemoryDomainId,
  type LocalModelCatalogCandidate,
  type LocalModelRecommendation,
  type LocalModelRecommendationProgressStep,
} from "@magnitudedev/sdk"
export { buildLocalInferenceSelections } from "@magnitudedev/client-common"
export type { LocalInferenceSelection } from "@magnitudedev/client-common"
import type { LocalInferenceSelection } from "@magnitudedev/client-common"

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
