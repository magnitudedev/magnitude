import type { Option } from "effect"
import type {
  LocalInferenceFitClass,
  LocalInferenceQuantization,
  LocalModelRecommendation as RecipeRecommendation,
  ProviderModelAvailability,
  ProviderModelId,
} from "@magnitudedev/sdk"

export type LocalModelRecommendation = RecipeRecommendation

export type LocalModelFitAssessment =
  | { readonly _tag: "NotAssessed" }
  | {
      readonly _tag: "Assessed"
      readonly requiredTotalBytes: number
      readonly domains: readonly {
        readonly memoryDomainId: string
        readonly requiredBytes: number
        readonly stableCapacityBytes: number
        readonly marginBytes: number
      }[]
      readonly result: "fits" | "does_not_fit"
    }

export interface LocalInferenceResidentMemoryDomain {
  readonly memoryDomainId: string
  readonly modelBytes: number
  readonly contextBytes: number
  readonly computeBytes: number
  readonly auxiliaryBytes: number
}

export interface LocalInferenceResidentMemory {
  readonly modelId: string
  readonly runtimeGeneration: number
  readonly domains: readonly LocalInferenceResidentMemoryDomain[]
}

export interface LocalInferenceHostProfile {
  readonly platform: string
  readonly architecture: string
  readonly topologyFingerprint: string
  readonly systemMemoryBytes: number
  readonly cpuModel: string | null
  readonly logicalCores: number
  readonly memoryDomains: readonly {
    readonly id: string
    readonly kind: "system" | "physical_device" | "unified_memory"
    readonly totalCapacityBytes: number
    readonly stableCapacityBytes: number
    readonly currentFreeBytes: number | null
    readonly sharesSystemMemory: boolean
    readonly backendNames: readonly string[]
    readonly deviceNames: readonly string[]
    readonly splitGroupId: string | null
  }[]
  readonly residentMemory: LocalInferenceResidentMemory | null
}

interface LocalModelChoiceFields {
  readonly choiceId: string
  readonly displayName: string
  readonly providerModelId: ProviderModelId
  readonly contextTokens: Option.Option<number>
  readonly fitClass: LocalInferenceFitClass
  readonly availability: ProviderModelAvailability
  readonly fitAssessment: LocalModelFitAssessment
  readonly explanation: string
  readonly residency: "loaded" | "sleeping" | "unloaded" | "loading" | "failed"
  readonly quantization: Option.Option<LocalInferenceQuantization>
  readonly sizeBytes: Option.Option<number>
}

export type LocalModelChoice =
  | Readonly<{ _tag: "Running" } & LocalModelChoiceFields>
  | Readonly<{ _tag: "Stored" } & LocalModelChoiceFields>

export type LocalInferenceOperationProgress =
  | { readonly completedBytes: number; readonly totalBytes: number }
  | { readonly fraction: number }

export interface LocalInferenceOperationSnapshot {
  readonly operationId: string
  readonly kind: "download" | "activate"
  readonly selectionId: string
  readonly providerModelId: ProviderModelId
  readonly status: "running" | "failed"
  readonly stage:
    | "queued"
    | "resolving"
    | "checking_space"
    | "downloading"
    | "publishing"
    | "assessing"
    | "unloading"
    | "loading"
    | "verifying"
    | "ready"
  readonly progress: Option.Option<LocalInferenceOperationProgress>
  readonly failure: Option.Option<{
    readonly code: string
    readonly message: string
    readonly retryable: boolean
  }>
  readonly startedAt: string
  readonly updatedAt: string
}

export type LocalInferenceRecommendationState =
  | { readonly _tag: "Loading" }
  | { readonly _tag: "Ready"; readonly recommendations: readonly LocalModelRecommendation[] }
  | { readonly _tag: "Failed"; readonly message: string }

export interface LocalInferenceState {
  readonly activeBinding: {
    readonly selectionId: string
    readonly providerModelId: ProviderModelId
    readonly contextTokens: number
  } | null
  readonly host: LocalInferenceHostProfile
  readonly choices: readonly LocalModelChoice[]
  readonly operations: readonly LocalInferenceOperationSnapshot[]
  readonly recommendationState: LocalInferenceRecommendationState
  readonly warnings: readonly { readonly code: string; readonly message: string }[]
}
