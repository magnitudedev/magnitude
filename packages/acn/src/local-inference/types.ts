import type {
  LocalInferenceCapabilities,
  LocalInferenceFitClass,
  LocalInferenceServingProfile,
  LocalModelChoice,
  LocalModelDownloadProgress,
} from "@magnitudedev/protocol"

export interface StableAcceleratorDomain {
  readonly memoryDomainId: string
  readonly capacityBytes: number
  readonly sharesSystemMemory: boolean | "unknown"
  readonly preferredBackend: string
  readonly modelSplitGroupId?: string
}

/**
 * Deliberately excludes all point-in-time free/available memory fields.
 * Keeping this as a separate type makes transient load structurally
 * unavailable to recommendation and ranking code.
 */
export interface StableInferenceCapacity {
  readonly systemMemoryBytes: number
  readonly acceleratorDomains: readonly StableAcceleratorDomain[]
}

export type QuantBitsClass = "q4" | "q5" | "q6" | "q8" | "fp8" | "mxfp4" | "other"

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
  /** Contexts supported by the model and covered by the conservative estimator. */
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
    readonly bitsClass: QuantBitsClass
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
  /** Curated relative rank based on model-specific coding/tool evidence. */
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

export interface LlamaCppRuntimeReadiness {
  readonly status: "ready" | "integration_pending" | "error"
  readonly canDownload: boolean
  readonly canActivate: boolean
  readonly diagnostic?: string
}

export interface LlamaCppRuntimeInventory {
  readonly running: readonly LocalModelChoice[]
  readonly downloaded: readonly LocalModelChoice[]
}

export interface LlamaCppActivationResult {
  readonly providerId: string
  readonly providerModelId: string
  readonly contextTokens: number
  readonly parallelSlots?: number
}

export interface LlamaCppHuggingFaceSource {
  readonly catalogModelId: string
  readonly configurationId: string
  readonly repo: string
  readonly revision: string
  readonly quantTag: string
  readonly contextTokens: number
  readonly servingProfile: LocalInferenceServingProfile
  readonly expectedFiles: readonly {
    readonly path: string
    readonly sizeBytes: number
    readonly sha256: string
  }[]
}

export interface LlamaCppRuntimeBridgeShape {
  readonly getReadiness: import("effect").Effect.Effect<LlamaCppRuntimeReadiness, import("@magnitudedev/protocol").SessionError>
  readonly getCapabilities: import("effect").Effect.Effect<LocalInferenceCapabilities, import("@magnitudedev/protocol").SessionError>
  readonly getInventory: import("effect").Effect.Effect<LlamaCppRuntimeInventory, import("@magnitudedev/protocol").SessionError>
  readonly startDownload: (
    source: LlamaCppHuggingFaceSource,
  ) => import("effect").Effect.Effect<{ readonly operationId: string }, import("@magnitudedev/protocol").SessionError>
  readonly subscribeDownload: (
    operationId: string,
  ) => import("effect").Stream.Stream<LocalModelDownloadProgress, import("@magnitudedev/protocol").SessionError>
  readonly cancelDownload: (
    operationId: string,
  ) => import("effect").Effect.Effect<void, import("@magnitudedev/protocol").SessionError>
  readonly activate: (
    selection: LocalModelChoice | EvaluatedLocalConfiguration,
  ) => import("effect").Effect.Effect<LlamaCppActivationResult, import("@magnitudedev/protocol").SessionError>
}
