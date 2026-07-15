// ── Errors ──
export * as LlamaCppErrors from "./errors"
export {
  LlamaCppBinaryNotFound,
  LlamaCppBinaryVersionTooOld,
  LlamaCppBinaryDownloadFailed,
  LlamaCppBinaryValidationFailed,
  LlamaCppUnsupportedPlatform,
  LlamaCppHardwareError,
  LlamaCppModelNotFound,
  LlamaCppModelDownloadFailed,
  LlamaCppGatedModelAccessDenied,
  LlamaCppHfTokenMissing,
  LlamaCppServerStartFailed,
  LlamaCppServerTimeout,
  LlamaCppServerOutOfMemory,
  LlamaCppEndpointError,
  LlamaCppPortUnavailable,
  LlamaCppDetectionFailed,
} from "./errors"

// ── Version ──
export {
  MINIMUM_LLMACPP_VERSION,
  RECOMMENDED_LLMACPP_VERSION,
  meetsMinimum,
  parseVersionNumber,
  buildNumberToTag,
  tagToBuildNumber,
} from "./version"

// ── Paths ──
export {
  llamacppDataDir,
  cachedBinaryDir,
  cachedBinaryPath,
  versionMarkerPath,
  downloadTmpDir,
  presetDir,
} from "./paths"

// ── Platform ──
export {
  detectPlatform,
  realArch,
  assetName,
  downloadUrl,
  GpuPreference,
  PlatformAsset,
  PlatformInfo,
} from "./platform"

// ── Binary ──
export {
  LlamaCppBinary,
  makeLlamaCppBinary,
  type LlamaCppBinaryApi,
  type LlamaCppBinaryDeps,
} from "./binary/resolve"

export { validateBinary } from "./binary/validate"
export { downloadBinary } from "./binary/download"
export { BinarySource, ResolvedBinary, BinaryStatus, DownloadResult } from "./binary/types"

// ── Hardware ──
export {
  LlamaCppHardware,
  makeLlamaCppHardware,
  type LlamaCppHardwareApi,
} from "./hardware/detect"

export {
  computeLimits,
  categorizeFit,
  assessHeuristic,
  assessWithFitPrint,
  parseFitPrintOutput,
} from "./hardware/fit"

export { parseListDevicesOutput } from "./hardware/detect"

export { getCpuInfo, getMemoryInfo } from "./hardware/os-info"

export {
  ModelFitCategory,
  GpuDevice,
  CpuInfo,
  MemoryInfo,
  HardwareInfo,
  DevicePlacement,
  ModelFitAssessment,
  AssessModelFitParams,
} from "./hardware/types"

// ── Model Store ──
export {
  LlamaCppModelStore,
  makeLlamaCppModelStore,
  type LlamaCppModelStoreApi,
  type LlamaCppModelStoreDeps,
} from "./models/store"

export { scanHfCache, parseRepoFolder, hfCacheDir } from "./models/hf-cache"
export { scanDirectory } from "./models/scan"
export { readGgufMetadata } from "./models/gguf"
export { groupShards } from "./models/shard"
export { pairMmproj } from "./models/mmproj"
export { resolveHfToken } from "./models/hf-token"
export { makeDownloadRegistry, type DownloadRegistry } from "./models/download-registry"
export {
  listRepoGgufFiles,
  downloadModelStream,
  cancelModelDownload,
  hfCachePathForFile,
  incompletePath,
} from "./models/download"

export {
  LocalModelInfo,
  LocalModelSource,
  ExpandedGgufMetadata,
  ShardGroup,
  DownloadModelParams,
  DownloadModelResult,
  DownloadProgress,
  DownloadStatus,
  DownloadState,
  RepoGgufFile,
  DiscoverOptions,
  DownloadEvent,
} from "./models/types"

// ── Inference ──
export {
  LlamaCppInference,
  makeLlamaCppInference,
  type LlamaCppInferenceApi,
  type LlamaCppInferenceDeps,
} from "./inference/inference"

export {
  LlamaCppInstances,
  makeLlamaCppInstances,
  type LlamaCppInstancesApi,
  type LlamaCppInstancesDeps,
} from "./inference/instances"

export {
  ServerMode,
  InstanceHealth,
  InstanceCapabilities,
  InstanceModelStatus,
  InstanceModelRef,
  InstanceInfo,
  InstanceOptions,
  ModelAvailability,
  LoadType,
  LoadedModel,
  AvailableModel,
  EnsureModelOptions,
  PresetDefaults,
  DetectedServer,
} from "./inference/types"

export { fingerprintServer } from "./inference/fingerprint"
export { findFreePort } from "./inference/ports"
export { generatePreset, writePreset } from "./inference/router"
export { waitForReady, detectOom, parseListeningPort, OOM_PATTERNS } from "./inference/health"
export { LlamaCppDetector, makeLlamaCppDetector, type LlamaCppDetectorApi } from "./inference/detector"
export { LlamaCppServer, makeLlamaCppServer, type LlamaCppServerApi, type ServerHandle } from "./inference/server"
