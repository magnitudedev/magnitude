import type {
  LocalInferenceFitClass,
  LocalInferenceServingProfile,
} from "@magnitudedev/protocol"

export interface StableAcceleratorDomain {
  readonly memoryDomainId: string
  readonly capacityBytes: number
  readonly sharesSystemMemory: boolean
  readonly preferredBackend: string
  readonly modelSplitGroupId?: string
}

/** Stable capacity only; point-in-time free memory is structurally unavailable. */
export interface StableInferenceCapacity {
  readonly systemMemoryBytes: number
  readonly acceleratorDomains: readonly StableAcceleratorDomain[]
}

export interface LocalModelAttentionCacheGroup {
  readonly layerCount: number
  readonly keyHeads: number
  readonly keyLength: number
  readonly valueHeads: number
  readonly valueLength: number
  /** Sliding-attention groups retain at most this many tokens per slot. */
  readonly contextLimitTokens?: number
}

export interface LocalModelRuntimeMetadata {
  readonly ggufArchitecture: string
  readonly parameterCount: number
  readonly blockCount: number
  readonly embeddingLength: number
  readonly attentionHeadCount: number
  readonly attentionCache: readonly LocalModelAttentionCacheGroup[]
  readonly recurrentState?: {
    readonly layerCount: number
    readonly innerSize: number
    readonly stateSize: number
    readonly convolutionWidth: number
    readonly bytesPerElement: number
  }
}

export interface LocalModelCatalogEntry {
  readonly id: string
  readonly modelId: string
  readonly family: string
  readonly displayName: string
  readonly architecture: "dense" | "moe"
  readonly totalParametersBillions?: number
  readonly activeParametersBillions?: number
  readonly effectiveParametersBillions?: number
  readonly modelMaximumContextTokens: number
  readonly supportedContextTokens: readonly number[]
  readonly repo: string
  readonly revision: string
  readonly quantTag: string
  readonly runtime: LocalModelRuntimeMetadata
  readonly files: readonly {
    readonly path: string
    readonly sizeBytes: number
    readonly sha256: string
  }[]
  readonly quantization: {
    readonly format: string
    readonly quantAwareCheckpoint: boolean
    readonly fidelityRank: number
    readonly fidelityLabel: string
    readonly fidelityEvidence: string
    readonly fidelitySourceUrl: string
  }
  readonly license: {
    readonly id: string
    readonly url: string
    readonly acknowledgementRequired: boolean
  }
  readonly modelQualityRank: number
}

export interface EvaluatedLocalConfiguration {
  readonly entry: LocalModelCatalogEntry
  readonly configurationId: string
  readonly contextTokens: number
  readonly servingProfile: LocalInferenceServingProfile
  readonly estimatedRuntimeBytes: number
  readonly stableCapacityBudgetBytes: number
  readonly fitMarginBytes: number
  readonly fitClass: LocalInferenceFitClass
  readonly constrainedContext: boolean
}
