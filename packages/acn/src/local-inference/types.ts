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
