import { Schema } from "effect"
import { GpuPreference } from "../platform"
import { LocalModelInfo } from "../models/types"

// ── Instance types ──

/** Server mode: `router` (multi-model, hot-swap) or `single-model`. */
export const ServerMode = Schema.Literal("router", "single-model")
export type ServerMode = Schema.Schema.Type<typeof ServerMode>

/** Health state of a running instance (fresh-probed). */
export const InstanceHealth = Schema.Literal("healthy", "loading", "unhealthy")
export type InstanceHealth = Schema.Schema.Type<typeof InstanceHealth>

/** Capabilities of a running instance — what operations are possible. */
export const InstanceCapabilities = Schema.Struct({
  /** We spawned it — can stop/restart. */
  canManage: Schema.Boolean,
  /** Router mode — can load/unload models via API. */
  canHotSwap: Schema.Boolean,
})
export type InstanceCapabilities = Schema.Schema.Type<typeof InstanceCapabilities>

/** Lifecycle state of a model on a running instance. */
export const InstanceModelStatus = Schema.Literal(
  "loaded",
  "loading",
  "sleeping",
  "unloaded",
)
export type InstanceModelStatus = Schema.Schema.Type<typeof InstanceModelStatus>

/** A model reference on a running instance. */
export const InstanceModelRef = Schema.Struct({
  /** Model ID (alias or path). */
  id: Schema.String,
  /** Lifecycle state on the server. */
  status: InstanceModelStatus,
  /** Whether WE loaded this model (tracked for smart unload). */
  loadedByUs: Schema.Boolean,
  /** File path if known (from `/props` or `/v1/models` meta). */
  path: Schema.NullOr(Schema.String),
})
export type InstanceModelRef = Schema.Schema.Type<typeof InstanceModelRef>

/** Full information about a running llama.cpp server instance. */
export const InstanceInfo = Schema.Struct({
  /** Stable ID: `managed-<port>` or `adopted-<host>-<port>`. */
  id: Schema.String,
  /** HTTP endpoint URL. */
  endpoint: Schema.String,
  /** Port number. */
  port: Schema.Number,
  /** Server mode. */
  mode: ServerMode,
  /** Health state (fresh-probed). */
  health: InstanceHealth,
  /** Did we spawn this process? */
  managed: Schema.Boolean,
  /** Process PID (`null` for adopted if we can't determine it). */
  pid: Schema.NullOr(Schema.Number),
  /** What operations are possible on this instance. */
  capabilities: InstanceCapabilities,
  /** Models on this instance (from `GET /v1/models` or `GET /models`). */
  models: Schema.Array(InstanceModelRef),
  /** Build info from `/props`, if available. */
  buildInfo: Schema.NullOr(Schema.String),
})
export type InstanceInfo = Schema.Schema.Type<typeof InstanceInfo>

/** Options for restarting a managed instance. */
export const InstanceOptions = Schema.Struct({
  /** Context size override (0 = model default). */
  contextSize: Schema.optional(Schema.Number),
  /** GPU layers override (-1 = auto via `--fit`). */
  gpuLayers: Schema.optional(Schema.Number),
  /** Number of parallel slots. */
  parallelSlots: Schema.optional(Schema.Number),
  /** Preferred port (0 = auto-select). */
  port: Schema.optional(Schema.Number),
  /** Idle seconds before model is unloaded (sleep mode). */
  sleepIdleSeconds: Schema.optional(Schema.Number),
  /** GPU preference for build selection. */
  gpuPreference: Schema.optional(GpuPreference),
})
export type InstanceOptions = Schema.Schema.Type<typeof InstanceOptions>

// ── Inference types ──

/** Availability state of a model across all running instances. */
export const ModelAvailability = Schema.Literal(
  "loaded",
  "available",
  "loading",
  "sleeping",
)
export type ModelAvailability = Schema.Schema.Type<typeof ModelAvailability>

/** How a model ended up loaded on the serving instance. */
export const LoadType = Schema.Literal(
  "already-loaded",
  "hot-swapped",
  "server-started",
  "server-restarted",
)
export type LoadType = Schema.Schema.Type<typeof LoadType>

/** A model that is loaded and ready for inference on an instance. */
export const LoadedModel = Schema.Struct({
  /** HTTP endpoint URL for inference requests to this model. */
  endpoint: Schema.String,
  /** The model ID that was loaded. */
  modelId: Schema.String,
  /** Actual context size after `--fit` (from `GET /props` on the serving instance). */
  contextSize: Schema.Number,
  /** How the model ended up loaded. */
  loadType: LoadType,
  /** ID of the instance serving this model. */
  instanceId: Schema.String,
})
export type LoadedModel = Schema.Schema.Type<typeof LoadedModel>

/** A model in the unified availability list (disk + all running instances). */
export const AvailableModel = Schema.Struct({
  /** Model ID. */
  id: Schema.String,
  /** Human-readable display name. */
  displayName: Schema.String,
  /** Current availability state across all instances. */
  availability: ModelAvailability,
  /** Endpoint if loaded on an instance, `null` if only on disk. */
  endpoint: Schema.NullOr(Schema.String),
  /** Instance ID if loaded, `null` if only on disk. */
  instanceId: Schema.NullOr(Schema.String),
  /** Model metadata (from disk discovery or server `/v1/models`). */
  info: LocalModelInfo,
})
export type AvailableModel = Schema.Schema.Type<typeof AvailableModel>

/** Options for `ensureModelLoaded`. Only apply when starting a managed server. */
export const EnsureModelOptions = Schema.Struct({
  /** Context size override (0 = model default). */
  contextSize: Schema.optional(Schema.Number),
  /** GPU layers override (-1 = auto via `--fit`). */
  gpuLayers: Schema.optional(Schema.Number),
  /** Number of parallel slots. */
  parallelSlots: Schema.optional(Schema.Number),
  /** Preferred port (0 = auto-select). */
  port: Schema.optional(Schema.Number),
  /** Idle seconds before model is unloaded (sleep mode). */
  sleepIdleSeconds: Schema.optional(Schema.Number),
  /** GPU preference for build selection. */
  gpuPreference: Schema.optional(GpuPreference),
  /** Additional model IDs to preload in the router preset. */
  additionalModels: Schema.optional(Schema.Array(Schema.String)),
})
export type EnsureModelOptions = Schema.Schema.Type<typeof EnsureModelOptions>

// ── Preset types ──

/** Defaults for router mode preset generation. */
export const PresetDefaults = Schema.Struct({
  /** GPU layers (-1 for auto via `--fit`). */
  ngl: Schema.Number,
  /** Context size (0 for model default). */
  ctx: Schema.Number,
  /** Idle seconds before sleep. */
  sleepIdleSeconds: Schema.optional(Schema.Number),
  /** Model ID to preload on startup. */
  loadOnStartup: Schema.optional(Schema.String),
})
export type PresetDefaults = Schema.Schema.Type<typeof PresetDefaults>

// ── Detection types ──

/** Raw fingerprint data from probing a server endpoint. */
export const DetectedServer = Schema.Struct({
  /** HTTP endpoint URL. */
  endpoint: Schema.String,
  /** Port number parsed from the endpoint. */
  port: Schema.Number,
  /** Server mode. */
  mode: ServerMode,
  /** Models loaded on the server. */
  models: Schema.Array(InstanceModelRef),
  /** Build info string from `/props`. */
  buildInfo: Schema.NullOr(Schema.String),
})
export type DetectedServer = Schema.Schema.Type<typeof DetectedServer>
