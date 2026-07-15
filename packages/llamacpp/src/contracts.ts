import { Option, Secret } from "effect"

export interface LlamaCppReleaseAsset {
  readonly platform: "darwin" | "linux"
  readonly architecture: "arm64" | "x64"
  readonly accelerator: "metal" | "cpu" | "vulkan"
  readonly fileName: string
  readonly url: string
  readonly sizeBytes: number
  readonly sha256: string
}

export interface LlamaCppReleaseManifest {
  readonly build: number
  readonly tag: string
  readonly assets: readonly LlamaCppReleaseAsset[]
}

export interface LlamaCppDistributionConfig {
  readonly managedRoot: string
  readonly configuredExecutable?: string
  readonly accelerator?: "auto" | "cpu" | "vulkan"
  readonly release?: LlamaCppReleaseManifest
}

export interface ResolvedDistribution {
  readonly executablePath: string
  readonly directory: string
  readonly build: number
  readonly source: "managed" | "configured"
}

export type DistributionState =
  | { readonly _tag: "Missing" }
  | { readonly _tag: "UnsupportedPlatform"; readonly platform: string; readonly architecture: string }
  | { readonly _tag: "Invalid"; readonly reason: string }
  | { readonly _tag: "Ready"; readonly distribution: ResolvedDistribution }

export type DistributionInstallEvent =
  | { readonly _tag: "Resolving" }
  | { readonly _tag: "Downloading"; readonly completedBytes: number; readonly totalBytes: number }
  | { readonly _tag: "Extracting" }
  | { readonly _tag: "Verifying" }
  | { readonly _tag: "Publishing" }
  | { readonly _tag: "Ready"; readonly distribution: ResolvedDistribution }

export interface HostDevice {
  readonly backend: string
  readonly name: string
}

export interface LlamaCppMemoryDomain {
  readonly id: string
  readonly kind: "system" | "physical_device" | "unified_working_set"
  readonly stableCapacityBytes: number
  readonly currentFreeBytes: number | null
  readonly sharesSystemMemory: boolean
  readonly devices: readonly HostDevice[]
  readonly splitGroupId: string | null
}

export interface LlamaCppHostWarning {
  readonly code: string
  readonly message: string
}

export interface LlamaCppHostProfile {
  readonly system: {
    readonly totalMemoryBytes: number
    readonly cpuModel: string | null
    readonly logicalCores: number
  }
  readonly memoryDomains: readonly LlamaCppMemoryDomain[]
  readonly runtimeProbe: "not_installed" | "complete" | "partial"
  readonly warnings: readonly LlamaCppHostWarning[]
}

export interface ModelFitRequest {
  readonly modelBytes: number
  readonly contextBytesPerSlot: number
  readonly parallelSlots: number
  readonly modelLayerCount: number | null
}

export interface ModelFitPlan {
  readonly requiredBytes: number
  readonly stableCapacityBytes: number
  readonly parallelSlots: number
  readonly gpuLayers: number
  readonly splitMode: "none" | "layer"
  readonly fits: boolean
}

export type ModelArtifactSource =
  | { readonly _tag: "MagnitudeOwned"; readonly manifestId: string }
  | { readonly _tag: "HuggingFaceCache"; readonly repo: string; readonly revision: string }
  | { readonly _tag: "UserDirectory"; readonly directoryId: string }

export interface ModelArtifactMetadata {
  readonly displayName: string
  readonly architecture: string | null
  readonly quantization: string | null
  readonly contextLength: number | null
  readonly parameterCount: number | null
  readonly layerCount: number | null
  readonly tokenizerModel: string | null
  readonly tokenizerPre: string | null
  readonly baseModelNames: readonly string[]
}

export interface ModelArtifactSummary {
  readonly modelId: string
  readonly source: ModelArtifactSource
  readonly sizeBytes: number
  readonly metadata: ModelArtifactMetadata
  readonly hasVisionProjector: boolean
}

export interface ResolvedModelArtifact extends ModelArtifactSummary {
  readonly primaryPath: string
  readonly shardPaths: readonly string[]
  readonly projectorPath: string | null
}

export interface ModelStoreSnapshot {
  readonly artifacts: readonly ModelArtifactSummary[]
  readonly warnings: readonly { readonly code: string; readonly message: string }[]
}

export interface ArtifactDownloadFile {
  readonly path: string
  readonly sizeBytes: number
  readonly sha256: string
}

export interface ArtifactDownloadPlan {
  readonly artifactId: string
  readonly repo: string
  readonly revision: string
  readonly files: readonly ArtifactDownloadFile[]
  readonly safetyReserveBytes: number
}

export interface LlamaCppModelStoreConfig {
  readonly ownedRoot: string
  readonly huggingFaceCacheRoot?: string
  readonly userDirectories?: readonly {
    readonly directoryId: string
    readonly path: string
  }[]
  readonly huggingFaceToken?: Secret.Secret
}

export type ModelDownloadEvent =
  | { readonly _tag: "Resolving"; readonly artifactId: string }
  | { readonly _tag: "CheckingSpace"; readonly artifactId: string; readonly requiredBytes: number; readonly availableBytes: number }
  | { readonly _tag: "Downloading"; readonly artifactId: string; readonly file: string; readonly completedBytes: number; readonly totalBytes: number }
  | { readonly _tag: "Verifying"; readonly artifactId: string; readonly file: string }
  | { readonly _tag: "Publishing"; readonly artifactId: string }
  | { readonly _tag: "Ready"; readonly artifact: ModelArtifactSummary }

export interface LlamaCppConnection {
  readonly baseUrl: string
  readonly apiKey: Option.Option<Secret.Secret>
}

export interface VerifiedServedModelMetadata {
  readonly architecture: string | null
  readonly quantization: string | null
  readonly sizeBytes: number | null
}

export interface EnsureManagedServingRequest {
  readonly _tag: "Managed"
  readonly modelId: string
  readonly providerModelId: string
  readonly contextTokens: number
  readonly fitPlan: ModelFitPlan
}

export interface EnsureExternalServingRequest {
  readonly _tag: "External"
  readonly connectionId: string
  readonly providerModelId: string
  readonly contextTokens: number
}

export type EnsureServingRequest = EnsureManagedServingRequest | EnsureExternalServingRequest

export interface ServingTarget {
  readonly serverId: string
  readonly ownership: "managed" | "external"
  readonly providerModelId: string
  readonly configuredContextTokens: number
  readonly metadata: VerifiedServedModelMetadata
  readonly connection: LlamaCppConnection
}

export interface ServedModelObservation {
  readonly providerModelId: string
  readonly modelPath: string | null
  readonly displayName: string | null
  readonly contextTokens: number | null
  readonly quantization: string | null
  readonly sizeBytes: number | null
}

export interface LlamaCppServerObservation {
  readonly serverId: string
  readonly ownership: "managed" | "external"
  readonly health: "ready" | "loading" | "unhealthy"
  readonly models: readonly ServedModelObservation[]
  readonly build: string | null
}

export interface LlamaCppRuntimeSnapshot {
  readonly managed: LlamaCppServerObservation | null
  readonly external: readonly LlamaCppServerObservation[]
}

export interface LlamaCppRuntimeConfig {
  readonly runtimeRoot: string
  readonly externalConnections?: () => import("effect").Effect.Effect<readonly {
    readonly connectionId: string
    readonly connection: LlamaCppConnection
  }[]>
  readonly preferredPort?: number
}
