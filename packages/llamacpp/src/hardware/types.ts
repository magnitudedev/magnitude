import { Schema } from "effect"

/**
 * Model fit category relative to available hardware resources.
 *
 * - `"fully-accelerated"` — model fits entirely in GPU memory (fast).
 * - `"partial-cpu"` — model exceeds GPU memory but fits in GPU+RAM (slower, some offloading).
 * - `"wont-fit"` — model exceeds total available memory (cannot run).
 */
export const ModelFitCategory = Schema.Literal(
  "fully-accelerated",
  "partial-cpu",
  "wont-fit",
)
export type ModelFitCategory = Schema.Schema.Type<typeof ModelFitCategory>

/**
 * A GPU/accelerator device reported by `llama-server --list-devices`.
 */
export const GpuDevice = Schema.Struct({
  /** Backend identifier (e.g. `MTL`, `Vulkan`, `CUDA`). */
  backend: Schema.String,
  /** Human-readable device name. */
  name: Schema.String,
  /** Total device memory in bytes. */
  totalBytes: Schema.Number,
  /** Free device memory in bytes. */
  freeBytes: Schema.Number,
})
export type GpuDevice = Schema.Schema.Type<typeof GpuDevice>

/** CPU information obtained from `node:os`. */
export const CpuInfo = Schema.Struct({
  /** CPU model string. */
  model: Schema.String,
  /** Number of logical cores. */
  cores: Schema.Number,
})
export type CpuInfo = Schema.Schema.Type<typeof CpuInfo>

/** Memory information (total and available) in bytes. */
export const MemoryInfo = Schema.Struct({
  /** Total system memory in bytes. */
  totalBytes: Schema.Number,
  /** Currently available memory in bytes. */
  availableBytes: Schema.Number,
})
export type MemoryInfo = Schema.Schema.Type<typeof MemoryInfo>

/**
 * Complete hardware snapshot: CPU, memory, and GPU devices.
 * Used for model fit assessment.
 */
export const HardwareInfo = Schema.Struct({
  cpu: CpuInfo,
  memory: MemoryInfo,
  /** GPU/accelerator devices from `--list-devices` (empty if CPU-only). */
  gpus: Schema.Array(GpuDevice),
  /** True on Apple Silicon where GPU and CPU share unified memory. */
  isUnifiedMemory: Schema.Boolean,
})
export type HardwareInfo = Schema.Schema.Type<typeof HardwareInfo>

/**
 * Per-device memory placement from `--fit-print` output.
 * Shows how much model/context/compute memory would land on each device.
 */
export const DevicePlacement = Schema.Struct({
  /** Device identifier from fit-print output. */
  device: Schema.String,
  /** Model weights bytes on this device. */
  modelBytes: Schema.Number,
  /** Context (KV cache) bytes on this device. */
  contextBytes: Schema.Number,
  /** Compute buffer bytes on this device. */
  computeBytes: Schema.Number,
})
export type DevicePlacement = Schema.Schema.Type<typeof DevicePlacement>

/** Result of assessing whether a model fits on the current hardware. */
export const ModelFitAssessment = Schema.Struct({
  /** Which fit category the model falls into. */
  category: ModelFitCategory,
  /** Upper bound for fully-accelerated (GPU-only) placement. */
  fastLimitBytes: Schema.Number,
  /** Upper bound for any placement (GPU + RAM). */
  ceilingBytes: Schema.Number,
  /** Per-device placement from `--fit-print`, if available. */
  placement: Schema.optional(Schema.Array(DevicePlacement)),
})
export type ModelFitAssessment = Schema.Schema.Type<typeof ModelFitAssessment>

/** Parameters for model fit assessment. */
export const AssessModelFitParams = Schema.Struct({
  /** Current hardware snapshot. */
  hardware: HardwareInfo,
  /** Model file size in bytes (used for heuristic assessment). */
  modelSizeBytes: Schema.Number,
  /** Optional model path for precise `--fit-print` assessment. */
  modelPath: Schema.optional(Schema.String),
})
export type AssessModelFitParams = Schema.Schema.Type<typeof AssessModelFitParams>
