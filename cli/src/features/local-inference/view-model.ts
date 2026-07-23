import { Option } from "effect"
import {
  ProviderModelCatalogLifecycle,
  ProviderIdSchema,
  type LocalInferenceHardware,
  type LocalInferenceMemoryDomainId,
  type LocalModel,
  type LocalModelRecommendation,
  type LocalModelRecommendationProgressStep,
  type ProviderModelId,
  type ReasoningEffort,
} from "@magnitudedev/sdk"
import type { LocalInferenceView } from "@magnitudedev/client-common"

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")

export type LocalInferenceSelection = {
  readonly kind: "running" | "stored" | "recommendation"
  readonly id: string
  readonly model: LocalModel
  readonly recommendation: Option.Option<LocalModelRecommendation>
  readonly providerModelId: Option.Option<ProviderModelId>
  readonly reasoningEffort: Option.Option<ReasoningEffort>
}

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
    ? Option.match(left.recommendation, {
        onNone: () => 4,
        onSome: ({ intent }) => recommendationIntentOrder[intent],
      }) - Option.match(right.recommendation, {
        onNone: () => 4,
        onSome: ({ intent }) => recommendationIntentOrder[intent],
      })
    : 0)
  || left.model.displayName.localeCompare(right.model.displayName)

export const buildLocalInferenceSelections = (
  view: LocalInferenceView,
): readonly LocalInferenceSelection[] => {
  const running = new Set([view.slots.slots.primary, view.slots.slots.secondary].flatMap((slot) =>
    slot._tag === "Ready" && slot.selection.providerId === LOCAL_PROVIDER_ID
      ? [slot.selection.providerModelId]
      : []))
  const catalogModels = ProviderModelCatalogLifecycle.match(view.catalog, {
    Loading: () => [],
    Ready: ({ models }) => models,
    Refreshing: ({ models }) => models,
    Degraded: ({ models }) => models,
    Unavailable: () => [],
  })
  const localProviderIds = new Set(catalogModels
    .filter(({ providerId, availability }) =>
      providerId === LOCAL_PROVIDER_ID && availability._tag === "Available")
    .map(({ providerModelId }) => providerModelId))
  const stored = view.models.models
    .filter(({ download }) => download._tag === "Downloaded")
    .map((model): LocalInferenceSelection => {
      const providerModelId = model.preparation._tag === "Available"
        ? Option.fromNullable(model.preparation.providerModelIds.find((id) => localProviderIds.has(id)))
        : Option.none<ProviderModelId>()
      const providerModel = Option.flatMap(providerModelId, (id) =>
        Option.fromNullable(catalogModels.find(({ providerModelId }) => providerModelId === id)))
      return {
        id: `model:${model.id}`,
        kind: Option.exists(providerModelId, (id) => running.has(id)) ? "running" : "stored",
        model,
        recommendation: Option.none(),
        providerModelId,
        reasoningEffort: Option.flatMap(
          providerModel,
          ({ capabilities }) => capabilities.reasoning.defaultEffort,
        ),
      }
    })
  const recommendations = view.models.recommendations._tag === "Ready"
    ? view.models.recommendations.entries.flatMap((recommendation): readonly LocalInferenceSelection[] => {
        const model = view.models.models.find(({ id }) => id === recommendation.modelId)
        if (!model || model.download._tag === "Downloaded") return []
        return [{
          id: `recommendation:${recommendation.id}`,
          kind: "recommendation",
          model,
          recommendation: Option.some(recommendation),
          providerModelId: Option.none(),
          reasoningEffort: Option.none(),
        }]
      })
    : []
  const representedModelIds = new Set(recommendations.map(({ model }) => model.id))
  const transientDownloads = view.models.models
    .filter((model) =>
      (model.download._tag === "Downloading" || model.download._tag === "Failed")
      && !representedModelIds.has(model.id))
    .map((model): LocalInferenceSelection => ({
      id: `download:${model.id}`,
      kind: "recommendation",
      model,
      recommendation: Option.none(),
      providerModelId: Option.none(),
      reasoningEffort: Option.none(),
    }))
  return [...stored, ...recommendations, ...transientDownloads].sort(compareSelections)
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

export const formatContext = (tokens: number): string => tokens < 1_000
  ? String(tokens)
  : tokens % 1_024 === 0
    ? `${tokens / 1_024}K`
    : `${Math.round(tokens / 1_000)}K`

export const formatModelLoadProgress = (percentage: number): string => `Loading ${percentage}%`

const progressLabel = (
  id: LocalModelRecommendationProgressStep["id"],
  completed: boolean,
): string => {
  if (id === "hardware") return completed ? "Detected hardware" : "Detecting your hardware"
  if (id === "catalog") {
    return completed ? "Loaded models from Hugging Face" : "Loading models from Hugging Face"
  }
  if (id === "metadata") {
    return completed ? "Checked model files" : "Checking model files"
  }
  if (id === "assessment") {
    return completed ? "Evaluated models for this machine" : "Evaluating models for this machine"
  }
  return completed ? "Prepared recommendations" : "Choosing recommendations"
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
  nowMs: number,
): readonly LocalInferenceProgressLine[] => steps.map((step) => {
  const completed = step.status._tag === "Completed"
  const label = progressLabel(step.id, completed)
  const count = Option.match(step.totalItems, {
    onNone: () => "",
    onSome: (total) => Option.match(step.completedItems, {
      onNone: () => ` · ${total}`,
      onSome: (value) => ` · ${value}/${total}`,
    }),
  })
  if (step.status._tag === "Pending") {
    return { id: step.id, state: "pending", label, metadata: "" }
  }
  if (step.status._tag === "Running") {
    return {
      id: step.id,
      state: "running",
      label,
      metadata: `${count} · ${formatDurationMs(Math.max(0, nowMs - step.status.startedAtMs))}`,
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
    metadata: `${count}${step.status.cached ? " · cached" : ` · ${formatDurationMs(step.status.durationMs)}`}`,
  }
})

export interface LocalHardwarePresentation {
  readonly system: { readonly name: string; readonly details: readonly string[] }
  readonly accelerators: readonly { readonly name: string; readonly details: string }[]
}

const unique = (values: readonly string[]): string[] =>
  [...new Set(values.map((value) => value.trim()).filter(Boolean))]

export const describeLocalHardware = (
  hardware: LocalInferenceHardware,
): LocalHardwarePresentation => {
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
  const isAppleSilicon =
    hardware.platform === "MacOS" && hardware.architecture === "Arm64"
  const processorName = Option.getOrElse(hardware.processor, () =>
    isAppleSilicon ? "Apple Silicon" : "CPU")
  const productName = Option.getOrElse(hardware.productName, () => "")
  const acceleratorName = unifiedAcceleratorNames.join(" + ")
  const name = isAppleSilicon
    ? processorName
    : unique([productName, acceleratorName]).join(" · ") || processorName
  return {
    system: {
      name,
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

export const selectionMetadata = ({ model, recommendation }: LocalInferenceSelection): string =>
  `${model.quantization} · ${formatBytes(model.downloadBytes)} · ${formatContext(
    Option.match(recommendation, {
      onNone: () => model.maximumContextLength,
      onSome: ({ profile }) => profile.contextLength,
    }),
  )} context`

export const selectionCapacityWarning = ({ model }: LocalInferenceSelection): string | null =>
  model.preparation._tag === "Unavailable" ? model.preparation.failure.message : null
