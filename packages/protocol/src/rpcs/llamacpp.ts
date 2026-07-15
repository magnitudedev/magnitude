import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SessionError } from "../errors"

// ── Schemas (wire types — match @magnitudedev/llamacpp exports) ──

export const LlamaCppBinaryStatusSchema = Schema.Struct({
  installed: Schema.Boolean,
  buildNumber: Schema.NullOr(Schema.Number),
  path: Schema.NullOr(Schema.String),
  source: Schema.NullOr(Schema.Literal(
    "env", "config", "cache", "path", "common-location", "download",
  )),
  meetsMinimum: Schema.Boolean,
  minimumRequired: Schema.Number,
  recommended: Schema.Number,
})

export const LocalModelInfoSchema = Schema.Struct({
  id: Schema.String,
  displayName: Schema.String,
  filePath: Schema.String,
  shardPaths: Schema.optional(Schema.Array(Schema.String)),
  mmprojPath: Schema.optional(Schema.String),
  architecture: Schema.optional(Schema.String),
  quantization: Schema.optional(Schema.String),
  contextLength: Schema.optional(Schema.Number),
  fileSizeBytes: Schema.Number,
  parameterCount: Schema.optional(Schema.Number),
  hiddenSize: Schema.optional(Schema.Number),
  layerCount: Schema.optional(Schema.Number),
  headCount: Schema.optional(Schema.Number),
  vocabSize: Schema.optional(Schema.Number),
  feedForwardLength: Schema.optional(Schema.Number),
  expertCount: Schema.optional(Schema.Number),
  expertUsedCount: Schema.optional(Schema.Number),
  tokenizerModel: Schema.optional(Schema.String),
  vision: Schema.Boolean,
  audio: Schema.Boolean,
  moe: Schema.Boolean,
  source: Schema.Union(
    Schema.TaggedStruct("hf-cache", { repoId: Schema.String, commit: Schema.String }),
    Schema.TaggedStruct("user-dir", { dir: Schema.String }),
  ),
  repoId: Schema.optional(Schema.String),
  commit: Schema.optional(Schema.String),
  baseModelNames: Schema.optional(Schema.Array(Schema.String)),
})

export const DownloadModelParamsSchema = Schema.Struct({
  repo: Schema.String,
  file: Schema.String,
  revision: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})

export const DownloadModelResultSchema = Schema.Struct({
  filePath: Schema.String,
  repoId: Schema.String,
  commit: Schema.String,
})

export const DownloadProgressSchema = Schema.Struct({
  downloadedBytes: Schema.Number,
  totalBytes: Schema.Number,
  percent: Schema.Number,
  bytesPerSecond: Schema.Number,
  etaSeconds: Schema.Number,
})

export const DownloadStateSchema = Schema.Struct({
  id: Schema.String,
  repo: Schema.String,
  file: Schema.String,
  status: Schema.Literal("downloading", "paused", "completed", "failed"),
  downloadedBytes: Schema.Number,
  totalBytes: Schema.Number,
  percent: Schema.Number,
  bytesPerSecond: Schema.Number,
  etaSeconds: Schema.Number,
  error: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})

export const RepoGgufFileSchema = Schema.Struct({
  path: Schema.String,
  size: Schema.Number,
  quantization: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})

export const GpuDeviceSchema = Schema.Struct({
  backend: Schema.String,
  name: Schema.String,
  totalBytes: Schema.Number,
  freeBytes: Schema.Number,
})

export const HardwareInfoSchema = Schema.Struct({
  cpu: Schema.Struct({
    model: Schema.String,
    cores: Schema.Number,
  }),
  memory: Schema.Struct({
    totalBytes: Schema.Number,
    availableBytes: Schema.Number,
  }),
  gpus: Schema.Array(GpuDeviceSchema),
  isUnifiedMemory: Schema.Boolean,
})

export const InstanceModelRefSchema = Schema.Struct({
  id: Schema.String,
  status: Schema.Literal("loaded", "loading", "sleeping", "unloaded"),
  loadedByUs: Schema.Boolean,
  path: Schema.NullOr(Schema.String),
})

export const InstanceCapabilitiesSchema = Schema.Struct({
  canManage: Schema.Boolean,
  canHotSwap: Schema.Boolean,
})

export const InstanceInfoSchema = Schema.Struct({
  id: Schema.String,
  endpoint: Schema.String,
  port: Schema.Number,
  mode: Schema.Literal("router", "single-model"),
  health: Schema.Literal("healthy", "loading", "unhealthy"),
  managed: Schema.Boolean,
  pid: Schema.NullOr(Schema.Number),
  capabilities: InstanceCapabilitiesSchema,
  models: Schema.Array(InstanceModelRefSchema),
  buildInfo: Schema.NullOr(Schema.String),
})

export const AvailableModelSchema = Schema.Struct({
  id: Schema.String,
  displayName: Schema.String,
  availability: Schema.Literal("loaded", "available", "loading", "sleeping"),
  endpoint: Schema.NullOr(Schema.String),
  instanceId: Schema.NullOr(Schema.String),
  info: LocalModelInfoSchema,
})

export const LoadedModelSchema = Schema.Struct({
  endpoint: Schema.String,
  modelId: Schema.String,
  contextSize: Schema.Number,
  loadType: Schema.Literal("already-loaded", "hot-swapped", "server-started", "server-restarted"),
  instanceId: Schema.String,
})

export const EnsureModelOptionsSchema = Schema.Struct({
  contextSize: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  gpuLayers: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  parallelSlots: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  port: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  sleepIdleSeconds: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  gpuPreference: Schema.optionalWith(Schema.Literal("auto", "cpu", "vulkan"), { as: "Option", exact: true }),
  additionalModels: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
})

export const InstanceOptionsSchema = Schema.Struct({
  contextSize: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  gpuLayers: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  parallelSlots: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  port: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  sleepIdleSeconds: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  gpuPreference: Schema.optionalWith(Schema.Literal("auto", "cpu", "vulkan"), { as: "Option", exact: true }),
})

// ── RPCs ──

// Binary
export const GetLlamaCppBinaryStatus = Rpc.make("GetLlamaCppBinaryStatus", {
  payload: Schema.Struct({}),
  success: LlamaCppBinaryStatusSchema,
  error: SessionError,
})

export const InstallLlamaCppBinary = Rpc.make("InstallLlamaCppBinary", {
  payload: Schema.Struct({}),
  success: LlamaCppBinaryStatusSchema,
  error: SessionError,
})

// Hardware
export const GetLlamaCppHardware = Rpc.make("GetLlamaCppHardware", {
  payload: Schema.Struct({}),
  success: HardwareInfoSchema,
  error: SessionError,
})

export const AssessLlamaCppModelFit = Rpc.make("AssessLlamaCppModelFit", {
  payload: Schema.Struct({
    modelSizeBytes: Schema.Number,
    modelPath: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({
    category: Schema.Literal("fully-accelerated", "partial-cpu", "wont-fit"),
    fastLimitBytes: Schema.Number,
    ceilingBytes: Schema.Number,
  }),
  error: SessionError,
})

// Models (ModelStore)
export const ListLocalModels = Rpc.make("ListLocalModels", {
  payload: Schema.Struct({
    extraDirs: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
  }),
  success: Schema.Struct({ models: Schema.Array(LocalModelInfoSchema) }),
  error: SessionError,
})

export const DeleteLlamaCppModel = Rpc.make("DeleteLlamaCppModel", {
  payload: Schema.Struct({ modelId: Schema.String }),
  success: Schema.Struct({}),
  error: SessionError,
})

export const ListLlamaCppRepoFiles = Rpc.make("ListLlamaCppRepoFiles", {
  payload: Schema.Struct({ repo: Schema.String }),
  success: Schema.Struct({ files: Schema.Array(RepoGgufFileSchema) }),
  error: SessionError,
})

export const DownloadLlamaCppModel = Rpc.make("DownloadLlamaCppModel", {
  payload: DownloadModelParamsSchema,
  success: Schema.Union(DownloadProgressSchema, DownloadModelResultSchema),
  error: SessionError,
  stream: true,
})

export const CancelLlamaCppDownload = Rpc.make("CancelLlamaCppDownload", {
  payload: DownloadModelParamsSchema,
  success: Schema.Struct({}),
  error: SessionError,
})

export const ListLlamaCppDownloads = Rpc.make("ListLlamaCppDownloads", {
  payload: Schema.Struct({}),
  success: Schema.Struct({ downloads: Schema.Array(DownloadStateSchema) }),
  error: SessionError,
})

// Inference
export const ListAvailableModels = Rpc.make("ListAvailableModels", {
  payload: Schema.Struct({}),
  success: Schema.Struct({ models: Schema.Array(AvailableModelSchema) }),
  error: SessionError,
})

export const EnsureLlamaCppModelLoaded = Rpc.make("EnsureLlamaCppModelLoaded", {
  payload: Schema.Struct({
    modelId: Schema.String,
    options: Schema.optionalWith(EnsureModelOptionsSchema, { as: "Option", exact: true }),
  }),
  success: LoadedModelSchema,
  error: SessionError,
})

export const UnloadLlamaCppModel = Rpc.make("UnloadLlamaCppModel", {
  payload: Schema.Struct({ modelId: Schema.String }),
  success: Schema.Struct({}),
  error: SessionError,
})

// Instances
export const ListLlamaCppInstances = Rpc.make("ListLlamaCppInstances", {
  payload: Schema.Struct({}),
  success: Schema.Struct({ instances: Schema.Array(InstanceInfoSchema) }),
  error: SessionError,
})

export const StopLlamaCppInstance = Rpc.make("StopLlamaCppInstance", {
  payload: Schema.Struct({ instanceId: Schema.String }),
  success: Schema.Struct({}),
  error: SessionError,
})

export const RestartLlamaCppInstance = Rpc.make("RestartLlamaCppInstance", {
  payload: Schema.Struct({
    instanceId: Schema.String,
    options: Schema.optionalWith(InstanceOptionsSchema, { as: "Option", exact: true }),
  }),
  success: InstanceInfoSchema,
  error: SessionError,
})

export const StopAllManagedInstances = Rpc.make("StopAllManagedInstances", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: SessionError,
})
