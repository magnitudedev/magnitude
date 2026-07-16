import type {
  LocalInferenceFitClass,
  LocalInferenceServingProfile,
  LocalInferenceUsageSelection,
  LocalModelRecommendation,
} from "@magnitudedev/protocol"
import { LOCAL_MODEL_CATALOG, catalogFileUrl, catalogSourcePageUrl } from "./catalog"
import type {
  EvaluatedLocalConfiguration,
  LocalModelCatalogEntry,
  StableInferenceCapacity,
} from "./types"

export const GIB = 1024 ** 3

export interface LlamaCppHostProfile {
  readonly system: { readonly totalMemoryBytes: number; readonly cpuModel?: string | null; readonly logicalCores?: number }
  readonly memoryDomains: readonly {
    readonly id: string
    readonly kind: "system" | "physical_device" | "unified_working_set"
    readonly stableCapacityBytes: number
    readonly sharesSystemMemory: boolean
    readonly splitGroupId?: string | null
    readonly currentFreeBytes?: number | null
    readonly devices: readonly { readonly backend: string; readonly name?: string }[]
  }[]
  readonly runtimeProbe?: unknown
  readonly warnings?: readonly unknown[]
}
export const BASELINE_CONTEXT_TOKENS = 32_768
export const MAIN_AGENT_CONTEXT_TARGETS = [200_000, 100_000] as const
export const SUBAGENT_CONTEXT_TARGETS = [100_000, 64_000] as const

export const configuredParallelSlots = (): number => {
  const configured = Number(process.env.MAGNITUDE_LLAMA_PARALLEL ?? 2)
  return Number.isSafeInteger(configured) && configured >= 1 && configured <= 4 ? configured : 2
}

export const parallelSlotsForUsage = (usage: LocalInferenceUsageSelection): number => {
  const requestsPerSession = usage.localModelRole === "main" ? 1 : 3
  const sessionMultiplier = usage.sessionConcurrency === "one" ? 1 : 3
  return requestsPerSession * sessionMultiplier
}

export const contextTargetsForUsage = (
  usage: LocalInferenceUsageSelection,
): typeof MAIN_AGENT_CONTEXT_TARGETS | typeof SUBAGENT_CONTEXT_TARGETS =>
  usage.localModelRole === "main" ? MAIN_AGENT_CONTEXT_TARGETS : SUBAGENT_CONTEXT_TARGETS

const minimumContextForUsage = (usage: LocalInferenceUsageSelection): number =>
  contextTargetsForUsage(usage).at(-1) ?? BASELINE_CONTEXT_TOKENS

const servingProfile = (
  entry: LocalModelCatalogEntry,
  usage: LocalInferenceUsageSelection,
  contextTokensPerSlot: number,
): LocalInferenceServingProfile => {
  const parallelSlots = parallelSlotsForUsage(usage)
  return {
    ...usage,
    parallelSlots,
    contextTokensPerSlot,
    totalContextCapacityTokens: parallelSlots * contextTokensPerSlot,
    slotAllocation: "uniform",
    runtimeProfileId: `conservative-v2:${entry.id}:p${parallelSlots}:ctx${contextTokensPerSlot}`,
  }
}

export const stableCapacityFromHost = (
  host: LlamaCppHostProfile,
): StableInferenceCapacity => {
  const domains = new Map<string, StableInferenceCapacity["acceleratorDomains"][number]>()
  for (const memory of host.memoryDomains) {
    if (memory.kind === "system" || memory.stableCapacityBytes <= 0) continue
    const current = domains.get(memory.id)
    const next = {
      memoryDomainId: memory.id,
      capacityBytes: memory.stableCapacityBytes,
      sharesSystemMemory: memory.sharesSystemMemory,
      preferredBackend: memory.devices[0]?.backend ?? "unknown",
      ...(memory.splitGroupId ? { modelSplitGroupId: memory.splitGroupId } : {}),
    } as const
    // Duplicate backend exposure of one physical domain is counted once.
    if (!current || next.capacityBytes < current.capacityBytes) domains.set(memory.id, next)
  }
  return {
    systemMemoryBytes: host.system.totalMemoryBytes,
    acceleratorDomains: [...domains.values()],
  }
}

export const systemCapacityBudget = (totalBytes: number): number =>
  Math.max(0, totalBytes - Math.max(8 * GIB, totalBytes * 0.2))

export const acceleratorCapacityBudget = (totalBytes: number): number =>
  Math.max(0, totalBytes - Math.max(GIB, totalBytes * 0.1))

/**
 * Conservative catalog estimate until measured llama.cpp profiles replace it.
 * It includes weights, compute/graph overhead, and context-scaled KV/runtime
 * overhead. The exact resolved artifact is still authoritatively fitted by
 * `LlamaCppHost.plan` immediately before activation.
 */
export const estimateRuntimeBytes = (
  entry: LocalModelCatalogEntry,
  contextTokensPerSlot: number,
  parallelSlots: number,
): number => estimateRuntimeBytesForModel(
  entry.files.reduce((total, file) => total + file.sizeBytes, 0),
  contextTokensPerSlot,
  parallelSlots,
)

export const estimateRuntimeBytesForModel = (
  modelBytes: number,
  contextTokensPerSlot: number,
  parallelSlots: number,
): number => {
  const computeAndGraph = Math.max(768 * 1024 ** 2, modelBytes * 0.04)
  const contextAt32K = Math.max(GIB, modelBytes * 0.06)
  const totalContextTokens = contextTokensPerSlot * parallelSlots
  const contextAndKv = contextAt32K * (totalContextTokens / BASELINE_CONTEXT_TOKENS)
  return Math.ceil(modelBytes + computeAndGraph + contextAndKv)
}

export const estimateRuntimeOverheadPerSlot = (
  modelBytes: number,
  contextTokensPerSlot: number,
  parallelSlots: number,
): number => Math.max(
  0,
  (estimateRuntimeBytesForModel(modelBytes, contextTokensPerSlot, parallelSlots) - modelBytes) / parallelSlots,
)

const fitConfiguration = (
  capacity: StableInferenceCapacity,
  entry: LocalModelCatalogEntry,
  usage: LocalInferenceUsageSelection,
  contextTokensPerSlot: number,
): EvaluatedLocalConfiguration | null => {
  const profile = servingProfile(entry, usage, contextTokensPerSlot)
  const estimatedRuntimeBytes = estimateRuntimeBytes(entry, contextTokensPerSlot, profile.parallelSlots)
  const systemBudget = systemCapacityBudget(capacity.systemMemoryBytes)
  const discreteDomains = capacity.acceleratorDomains
    .filter((domain) => domain.sharesSystemMemory === false)
  const discreteBudgets = discreteDomains.map((domain) => acceleratorCapacityBudget(domain.capacityBytes))
  const splitGroupBudgets = new Map<string, number>()
  for (const domain of discreteDomains) {
    if (!domain.modelSplitGroupId) continue
    splitGroupBudgets.set(
      domain.modelSplitGroupId,
      (splitGroupBudgets.get(domain.modelSplitGroupId) ?? 0) + acceleratorCapacityBudget(domain.capacityBytes),
    )
  }
  const unifiedBudgets = capacity.acceleratorDomains
    .filter((domain) => domain.sharesSystemMemory === true)
    .map((domain) => Math.min(systemBudget, domain.capacityBytes))
  // Multiple devices are combined only when the exact managed backend says
  // that they belong to a supported model-split group.
  const bestDiscrete = Math.max(0, ...discreteBudgets, ...splitGroupBudgets.values())
  const bestUnified = Math.max(0, ...unifiedBudgets)

  let fitClass: LocalInferenceFitClass | null = null
  let stableCapacityBudgetBytes = systemBudget
  if (bestUnified >= estimatedRuntimeBytes) {
    fitClass = "full_accelerator"
    stableCapacityBudgetBytes = Math.min(systemBudget, bestUnified)
  } else if (bestDiscrete >= estimatedRuntimeBytes && systemBudget >= 2 * GIB) {
    fitClass = "full_accelerator"
    stableCapacityBudgetBytes = bestDiscrete
  } else if (bestDiscrete > 0 && systemBudget + bestDiscrete >= estimatedRuntimeBytes && systemBudget >= 4 * GIB) {
    fitClass = "hybrid"
    stableCapacityBudgetBytes = systemBudget + bestDiscrete
  } else if (systemBudget >= estimatedRuntimeBytes) {
    fitClass = "cpu_or_unified"
  }
  if (!fitClass) return null

  return {
    entry,
    configurationId: `${entry.id}@${entry.revision}@role-${usage.localModelRole}@sessions-${usage.sessionConcurrency}@p-${profile.parallelSlots}@ctx-${contextTokensPerSlot}@${profile.runtimeProfileId}`,
    contextTokens: contextTokensPerSlot,
    servingProfile: profile,
    estimatedRuntimeBytes,
    stableCapacityBudgetBytes,
    fitMarginBytes: Math.max(0, stableCapacityBudgetBytes - estimatedRuntimeBytes),
    fitClass,
    constrainedContext: contextTokensPerSlot < minimumContextForUsage(usage),
  }
}

export const evaluateCatalog = (
  capacity: StableInferenceCapacity,
  usage: LocalInferenceUsageSelection,
  catalog: readonly LocalModelCatalogEntry[] = LOCAL_MODEL_CATALOG,
): EvaluatedLocalConfiguration[] => {
  const result: EvaluatedLocalConfiguration[] = []
  for (const entry of catalog) {
    for (const contextTokens of contextTargetsForUsage(usage)) {
      if (!entry.supportedContextTokens.includes(contextTokens)) continue
      if (contextTokens > entry.modelMaximumContextTokens) continue
      const evaluated = fitConfiguration(capacity, entry, usage, contextTokens)
      if (evaluated) result.push(evaluated)
    }
  }
  return result
}

const recommendedOrder = (a: EvaluatedLocalConfiguration, b: EvaluatedLocalConfiguration): number => {
  // Model quality first across models. Within one model, preserve useful
  // context before spending remaining memory on a marginally larger quant.
  if (a.entry.modelQualityRank !== b.entry.modelQualityRank) return b.entry.modelQualityRank - a.entry.modelQualityRank
  if (a.contextTokens !== b.contextTokens) return b.contextTokens - a.contextTokens
  if (a.entry.quantization.fidelityRank !== b.entry.quantization.fidelityRank) {
    return b.entry.quantization.fidelityRank - a.entry.quantization.fidelityRank
  }
  return b.fitMarginBytes - a.fitMarginBytes
}

const artifactBytes = (item: EvaluatedLocalConfiguration): number =>
  item.entry.files.reduce((total, file) => total + file.sizeBytes, 0)

/**
 * Curated one-step-down choices for tiers where raw byte minimization creates
 * an unhelpfully large capability jump. Values are model IDs, not artifacts,
 * so the policy still chooses the best fitting context and quant for that
 * smaller model on the detected machine.
 */
const CURATED_SMALLER_MODEL = new Map<string, string>([
  ["nemotron-3-super-120b-a12b", "qwen3.6-35b-a3b"],
  ["glm-5.2", "deepseek-v4-flash"],
])

/**
 * Context is a property of the selected model artifact, not a reason to pick a
 * different artifact. Collapse each model/quant to its largest fitting context
 * before comparing alternatives such as the lighter recommendation.
 */
const largestContextPerArtifact = (
  configurations: readonly EvaluatedLocalConfiguration[],
): EvaluatedLocalConfiguration[] => {
  const result = new Map<string, EvaluatedLocalConfiguration>()
  for (const item of configurations) {
    const current = result.get(item.entry.id)
    if (!current || item.contextTokens > current.contextTokens) {
      result.set(item.entry.id, item)
    }
  }
  return [...result.values()]
}

const toRecommendation = (
  item: EvaluatedLocalConfiguration,
  badge: LocalModelRecommendation["badge"],
): LocalModelRecommendation => {
  const entry = item.entry
  const totalDownloadBytes = entry.files.reduce((total, file) => total + file.sizeBytes, 0)
  const acceleration = item.fitClass === "full_accelerator"
    ? "expected to fit fully on a llama.cpp-visible accelerator"
    : item.fitClass === "hybrid"
      ? "expected to use conservative CPU/GPU hybrid placement"
      : "fits the stable system-memory budget for CPU or unified-memory inference"
  return {
    configurationId: item.configurationId,
    catalogModelId: entry.id,
    badge,
    displayName: entry.displayName,
    family: entry.family,
    architecture: entry.architecture,
    ...(entry.totalParametersBillions !== undefined ? { totalParametersBillions: entry.totalParametersBillions } : {}),
    ...(entry.activeParametersBillions !== undefined ? { activeParametersBillions: entry.activeParametersBillions } : {}),
    ...(entry.effectiveParametersBillions !== undefined
      ? { effectiveParametersBillions: entry.effectiveParametersBillions }
      : {}),
    quantization: {
      format: entry.quantization.format,
      quantAwareCheckpoint: entry.quantization.quantAwareCheckpoint,
      fidelityLabel: entry.quantization.fidelityLabel,
      fidelityEvidence: entry.quantization.fidelityEvidence,
      fidelitySourceUrl: entry.quantization.fidelitySourceUrl,
    },
    quantTag: entry.quantTag,
    repo: entry.repo,
    revision: entry.revision,
    files: entry.files.map((file) => ({
      ...file,
      downloadUrl: catalogFileUrl(entry, file.path),
    })),
    totalDownloadBytes,
    sourcePageUrl: catalogSourcePageUrl(entry),
    license: entry.license,
    contextTokens: item.contextTokens,
    servingProfile: item.servingProfile,
    modelMaximumContextTokens: entry.modelMaximumContextTokens,
    estimatedRuntimeBytes: item.estimatedRuntimeBytes,
    stableCapacityBudgetBytes: item.stableCapacityBudgetBytes,
    fitMarginBytes: item.fitMarginBytes,
    fitClass: item.fitClass,
    constrainedContext: item.constrainedContext,
    explanation: `${entry.quantization.fidelityLabel} ${entry.quantization.format} at ${Math.round(item.contextTokens / 1_000)}K context across ${item.servingProfile.parallelSlots} uniform local slot${item.servingProfile.parallelSlots === 1 ? "" : "s"}; ${acceleration}.`,
  }
}

export const recommendLocalModels = (
  capacity: StableInferenceCapacity,
  usage: LocalInferenceUsageSelection,
  catalog: readonly LocalModelCatalogEntry[] = LOCAL_MODEL_CATALOG,
): LocalModelRecommendation[] => {
  const evaluated = evaluateCatalog(capacity, usage, catalog).sort(recommendedOrder)
  const normal = evaluated.filter((item) => !item.constrainedContext)
  const pool = normal.length > 0 ? normal : evaluated
  const recommended = pool[0]
  if (!recommended) return []

  const choices: { item: EvaluatedLocalConfiguration; badge: LocalModelRecommendation["badge"] }[] = [
    { item: recommended, badge: "recommended" },
  ]

  const smallerPool = largestContextPerArtifact(pool)
    .filter((item) => item.entry.modelId !== recommended.entry.modelId)
    .filter((item) => artifactBytes(item) < artifactBytes(recommended))
  const curatedSmallerModelId = CURATED_SMALLER_MODEL.get(recommended.entry.modelId)
  const curatedSmaller = curatedSmallerModelId
    ? smallerPool
      .filter((item) => item.entry.modelId === curatedSmallerModelId)
      .sort(recommendedOrder)[0]
    : undefined
  const lighter = curatedSmaller ?? smallerPool
    .filter((item) => item.entry.modelQualityRank >= recommended.entry.modelQualityRank - 20)
    .sort((a, b) => artifactBytes(a) - artifactBytes(b) || recommendedOrder(a, b))[0]
  if (lighter) choices.push({ item: lighter, badge: "lighter" })

  const sameModelHigherFidelity = [...pool]
    .filter((item) => item.entry.id !== recommended.entry.id)
    .filter((item) => item.entry.modelId === recommended.entry.modelId)
    .filter((item) => item.entry.quantization.fidelityRank > recommended.entry.quantization.fidelityRank)
    .sort((a, b) => b.contextTokens - a.contextTokens || b.entry.quantization.fidelityRank - a.entry.quantization.fidelityRank)[0]
  if (sameModelHigherFidelity) choices.push({ item: sameModelHigherFidelity, badge: "higher_fidelity" })

  // Keep the selection set useful even when the primary artifact is
  // already the highest-fidelity quant. Prefer a third distinct model so the
  // cards represent real capability/weight trade-offs, then fall back to a
  // distinct quant only when fewer than three model families fit.
  while (choices.length < 3) {
    const chosenEntryIds = new Set(choices.map(({ item }) => item.entry.id))
    const chosenModelIds = new Set(choices.map(({ item }) => item.entry.modelId))
    const remaining = largestContextPerArtifact(pool)
      .filter((item) => !chosenEntryIds.has(item.entry.id))
      .sort(recommendedOrder)
    const alternative = remaining.find((item) => !chosenModelIds.has(item.entry.modelId)) ?? remaining[0]
    if (!alternative) break
    choices.push({ item: alternative, badge: "alternative" })
  }

  return choices.slice(0, 3).map(({ item, badge }) => toRecommendation(item, badge))
}

export const resolveConfiguration = (
  configurationId: string,
  capacity: StableInferenceCapacity,
  usage: LocalInferenceUsageSelection,
): EvaluatedLocalConfiguration | undefined =>
  evaluateCatalog(capacity, usage).find((configuration) => configuration.configurationId === configurationId)
