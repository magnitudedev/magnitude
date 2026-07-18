import type {
  LocalInferenceHostProfile,
  LocalInferenceState,
  LocalModelChoice,
  LocalModelRecommendation,
} from "@magnitudedev/sdk"

export type LocalInferenceSelection =
  | { readonly kind: "running" | "stored"; readonly id: string; readonly choice: LocalModelChoice }
  | { readonly kind: "recommendation"; readonly id: string; readonly recommendation: LocalModelRecommendation }

export const buildLocalInferenceSelections = (
  state: LocalInferenceState,
): LocalInferenceSelection[] => {
  const choices = state.choices
    .filter((choice) => choice.availability._tag === "Available")
    .map((choice): LocalInferenceSelection => ({
      kind: choice._tag === "RunningExternal" || choice._tag === "RunningManaged" ? "running" : "stored",
      id: choice.choiceId,
      choice,
    }))
  const choiceIds = new Set(choices.map((choice) => choice.id))
  return [
    ...choices,
    ...state.recommendations
      .filter((recommendation) => !choiceIds.has(recommendation.configurationId))
      .map((recommendation): LocalInferenceSelection => ({
        kind: "recommendation",
        id: recommendation.configurationId,
        recommendation,
      })),
  ]
}

export const selectedInferenceIndex = (
  selections: readonly LocalInferenceSelection[],
  selectedId: string | null,
): number => {
  const index = selectedId === null
    ? -1
    : selections.findIndex((selection) => selection.id === selectedId)
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

export interface LocalHardwarePresentation {
  readonly system: {
    readonly name: string
    readonly details: readonly string[]
  }
  readonly accelerators: readonly {
    readonly name: string
    readonly details: string
  }[]
}

const platformLabel = (platform: string): string => {
  if (platform === "darwin") return "macOS"
  if (platform === "linux") return "Linux"
  if (platform === "win32") return "Windows"
  return platform
}

const architectureLabel = (architecture: string): string => {
  if (architecture === "arm64") return "ARM64"
  if (architecture === "x64") return "x86-64"
  return architecture
}

const unique = (values: readonly string[]): string[] =>
  [...new Set(values.map((value) => value.trim()).filter(Boolean))]

const accelerationLabel = (backends: readonly string[]): string =>
  unique(backends).length > 0
    ? `${unique(backends).join(" + ")} GPU acceleration`
    : "GPU acceleration"

export const describeLocalHardware = (
  host: LocalInferenceHostProfile,
): LocalHardwarePresentation => {
  const accelerators = host.memoryDomains.filter((domain) => domain.kind !== "system")
  const unified = accelerators.filter((domain) => domain.kind === "unified_working_set")
  const discrete = accelerators.filter((domain) => domain.kind === "physical_device")
  const appleUnified = host.platform === "darwin"
    && host.architecture === "arm64"
    && unified.length > 0
  const systemName = host.cpuModel?.trim() || (
    host.platform === "darwin" && host.architecture === "arm64"
      ? "Apple Silicon"
      : "CPU"
  )
  const systemDetails = [
    `${platformLabel(host.platform)} · ${host.platform === "darwin" && host.architecture === "arm64" ? "Apple Silicon" : architectureLabel(host.architecture)} · ${host.logicalCores} logical CPU core${host.logicalCores === 1 ? "" : "s"}`,
    `${formatBytes(host.systemMemoryBytes)} ${appleUnified ? "unified" : "system"} memory${appleUnified ? ` · ${accelerationLabel(unified.flatMap((domain) => domain.backendNames))}` : ""}`,
  ]

  const visibleAccelerators = appleUnified ? discrete : accelerators
  return {
    system: { name: systemName, details: systemDetails },
    accelerators: visibleAccelerators.map((domain) => {
      const names = unique(domain.deviceNames)
      const backends = unique(domain.backendNames)
      const name = names.length > 0
        ? names.join(" + ")
        : domain.kind === "unified_working_set"
          ? "Integrated GPU"
          : `${backends[0] ?? "Local"} GPU`
      const memory = domain.kind === "unified_working_set"
        ? `${formatBytes(domain.totalCapacityBytes)} shared memory`
        : `${formatBytes(domain.totalCapacityBytes)} VRAM`
      return {
        name,
        details: `${memory} · ${accelerationLabel(backends)}`,
      }
    }),
  }
}

export const selectionTitle = (selection: LocalInferenceSelection): string =>
  selection.kind === "recommendation" ? selection.recommendation.displayName : selection.choice.displayName

export const selectionMetadata = (selection: LocalInferenceSelection): string => {
  const quantization = selection.kind === "recommendation"
    ? selection.recommendation.quantization.format
    : selection.choice.quantization?.format ?? "Quantization unavailable"
  const size = selection.kind === "recommendation"
    ? formatBytes(selection.recommendation.totalDownloadBytes)
    : selection.choice.sizeBytes === undefined
      ? "Size unavailable"
      : formatBytes(selection.choice.sizeBytes)
  const context = selection.kind === "recommendation"
    ? `${formatContext(selection.recommendation.contextTokens)} × ${selection.recommendation.servingProfile.parallelSlots} slot${selection.recommendation.servingProfile.parallelSlots === 1 ? "" : "s"}`
    : selection.choice.contextTokens === undefined
      ? "Context unavailable"
      : `${formatContext(selection.choice.contextTokens)} context`
  return `${quantization} · ${size} · ${context}`
}

export const selectionCapacityWarning = (selection: LocalInferenceSelection): string | null => {
  if (selection.kind === "recommendation") return null
  const fit = selection.choice.fitAssessment
  if (fit._tag !== "Estimated" || fit.result !== "capacity_risk") return null
  const constrained = fit.domains.filter(({ marginBytes }) => marginBytes < 0)
  const capacity = constrained.reduce((total, domain) => total + domain.stableCapacityBytes, 0)
  return capacity > 0
    ? `Memory warning: estimated ${formatBytes(fit.estimatedTotalBytes)}; constrained hardware capacity ${formatBytes(capacity)}. Loading may fail or affect system performance.`
    : `Memory warning: estimated ${formatBytes(fit.estimatedTotalBytes)} may exceed this machine's stable capacity.`
}
