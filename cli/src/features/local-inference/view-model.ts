import { Option } from "effect"
import {
  ProviderIdSchema,
  type LocalModelId,
  type LocalInferenceHardware,
  type LocalInferenceMemoryDomainId,
  type LocalModelInventoryEntry,
} from "@magnitudedev/sdk"
import type { LocalInferenceView } from "@magnitudedev/client-common"

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")

export type LocalInferenceSelection = {
  readonly kind: "running" | "stored" | "recommendation"
  readonly entry: LocalModelInventoryEntry
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
    ? Option.match(left.entry.model.recommendation, {
        onNone: () => 4,
        onSome: ({ intent }) => recommendationIntentOrder[intent],
      }) - Option.match(right.entry.model.recommendation, {
        onNone: () => 4,
        onSome: ({ intent }) => recommendationIntentOrder[intent],
      })
    : 0)
  || left.entry.model.displayName.localeCompare(right.entry.model.displayName)

export const buildLocalInferenceSelections = (
  view: LocalInferenceView,
): readonly LocalInferenceSelection[] => {
  if (view.inventory._tag !== "Ready") return []
  const running = new Set([view.slots.slots.primary, view.slots.slots.secondary].flatMap((slot) =>
    slot._tag === "Ready" && slot.selection.providerId === LOCAL_PROVIDER_ID
      ? [slot.selection.providerModelId]
      : []))
  return view.inventory.entries.map((entry): LocalInferenceSelection => ({
    kind: entry._tag === "Downloaded"
      ? running.has(entry.model.providerModelId) ? "running" : "stored"
      : "recommendation",
    entry,
  })).sort(compareSelections)
}

export const selectedInferenceIndex = (
  selections: readonly LocalInferenceSelection[],
  selectedId: Option.Option<LocalModelId>,
): number => {
  const index = Option.match(selectedId, {
    onNone: () => -1,
    onSome: (id) => selections.findIndex(({ entry }) => entry.model.localModelId === id),
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
  const name = Option.getOrElse(hardware.processor, () =>
    hardware.platform === "MacOS" && hardware.architecture === "Arm64" ? "Apple Silicon" : "CPU")
  const backendsFor = (memoryDomainId: LocalInferenceMemoryDomainId) => unique(hardware.accelerators
    .filter((accelerator) => accelerator.memoryDomainId === memoryDomainId)
    .map((accelerator) => accelerator.backend))
  const namesFor = (memoryDomainId: LocalInferenceMemoryDomainId) => unique(hardware.accelerators
    .filter((accelerator) => accelerator.memoryDomainId === memoryDomainId)
    .map((accelerator) => accelerator.name))
  const unifiedBackends = unique(unified.flatMap((domain) => backendsFor(domain.memoryDomainId)))
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

export const selectionTitle = ({ entry }: LocalInferenceSelection): string => entry.model.displayName

export const selectionMetadata = ({ entry }: LocalInferenceSelection): string =>
  `${entry.model.quantization} · ${formatBytes(entry.model.downloadBytes)} · ${formatContext(entry.model.contextWindow)} context`

export const selectionCapacityWarning = ({ entry }: LocalInferenceSelection): string | null =>
  entry.model.fit._tag === "DoesNotFit"
    ? `Memory warning: estimated ${formatBytes(entry.model.fit.requiredBytes)}; constrained hardware capacity ${formatBytes(entry.model.fit.availableBytes)}. Loading may fail or affect system performance.`
    : null
