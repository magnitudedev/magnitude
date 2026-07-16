import { createHash } from "node:crypto"
import { Data, Effect, Option, Schema } from "effect"
import { LlamaExecutionProfileId } from "./identity"
import { LlamaCliError } from "./cli-errors"

export const LlamaCacheType = Schema.Literal("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
export type LlamaCacheType = Schema.Schema.Type<typeof LlamaCacheType>

export const LlamaSplitMode = Schema.Literal("none", "layer", "row", "tensor")
export type LlamaSplitMode = Schema.Schema.Type<typeof LlamaSplitMode>

export type ContextSize = Data.TaggedEnum<{
  ModelDefault: Record<never, never>
  Tokens: { readonly value: number }
}>
export const ContextSize = Data.taggedEnum<ContextSize>()

export type OutputLimit = Data.TaggedEnum<{
  RuntimeDefault: Record<never, never>
  Tokens: { readonly value: number }
}>
export const OutputLimit = Data.taggedEnum<OutputLimit>()

export type GpuLayerSelection = Data.TaggedEnum<{
  Fit: Record<never, never>
  Exact: { readonly layers: number }
}>
export const GpuLayerSelection = Data.taggedEnum<GpuLayerSelection>()

export type FlashAttentionSelection = Data.TaggedEnum<{
  RuntimeDefault: Record<never, never>
  Enabled: Record<never, never>
  Disabled: Record<never, never>
}>
export const FlashAttentionSelection = Data.taggedEnum<FlashAttentionSelection>()

export type BatchSize = Data.TaggedEnum<{
  RuntimeDefault: Record<never, never>
  Exact: { readonly value: number }
}>
export const BatchSize = Data.taggedEnum<BatchSize>()

export type MicroBatchSize = Data.TaggedEnum<{
  RuntimeDefault: Record<never, never>
  Exact: { readonly value: number }
}>
export const MicroBatchSize = Data.taggedEnum<MicroBatchSize>()

export interface LlamaExecutionProfile {
  readonly id: LlamaExecutionProfileId
  readonly contextSize: ContextSize
  readonly outputLimit: OutputLimit
  readonly parallelSlots: number
  readonly gpuLayers: GpuLayerSelection
  readonly splitMode: LlamaSplitMode
  readonly tensorSplit: Option.Option<readonly number[]>
  readonly kvCache: { readonly key: LlamaCacheType; readonly value: LlamaCacheType }
  readonly flashAttention: FlashAttentionSelection
  readonly batchSize: BatchSize
  readonly microBatchSize: MicroBatchSize
  readonly mmap: boolean
  readonly mlock: boolean
}

const positiveInteger = (value: number): boolean => Number.isSafeInteger(value) && value > 0
const nonNegativeInteger = (value: number): boolean => Number.isSafeInteger(value) && value >= 0

const canonicalContextSize = ContextSize.$match({
  ModelDefault: () => "model-default",
  Tokens: ({ value }) => `tokens:${value}`,
})
const canonicalOutputLimit = OutputLimit.$match({
  RuntimeDefault: () => "runtime-default",
  Tokens: ({ value }) => `tokens:${value}`,
})
const canonicalGpuLayers = GpuLayerSelection.$match({
  Fit: () => "fit",
  Exact: ({ layers }) => `exact:${layers}`,
})
const canonicalFlashAttention = FlashAttentionSelection.$match({
  RuntimeDefault: () => "runtime-default",
  Enabled: () => "enabled",
  Disabled: () => "disabled",
})
const canonicalBatchSize = BatchSize.$match({
  RuntimeDefault: () => "runtime-default",
  Exact: ({ value }) => `exact:${value}`,
})
const canonicalMicroBatchSize = MicroBatchSize.$match({
  RuntimeDefault: () => "runtime-default",
  Exact: ({ value }) => `exact:${value}`,
})

const profileError = (field: string): LlamaCliError => LlamaCliError.make("profile", "invalid-input", Option.some(field))

export const makeLlamaExecutionProfile = (
  input: Omit<LlamaExecutionProfile, "id">,
): Effect.Effect<LlamaExecutionProfile, LlamaCliError> => Effect.gen(function* () {
  if (input.contextSize._tag === "Tokens" && !positiveInteger(input.contextSize.value)) return yield* profileError("contextSize")
  if (input.outputLimit._tag === "Tokens" && !positiveInteger(input.outputLimit.value)) return yield* profileError("outputLimit")
  if (!positiveInteger(input.parallelSlots)) return yield* profileError("parallelSlots")
  if (input.gpuLayers._tag === "Exact" && !nonNegativeInteger(input.gpuLayers.layers)) return yield* profileError("gpuLayers")
  if (Option.exists(input.tensorSplit, (values) => values.length === 0 || values.some((value) => !Number.isFinite(value) || value < 0))) return yield* profileError("tensorSplit")
  if (input.batchSize._tag === "Exact" && !positiveInteger(input.batchSize.value)) return yield* profileError("batchSize")
  if (input.microBatchSize._tag === "Exact" && !positiveInteger(input.microBatchSize.value)) return yield* profileError("microBatchSize")

  const canonical = [
    canonicalContextSize(input.contextSize),
    canonicalOutputLimit(input.outputLimit),
    input.parallelSlots,
    canonicalGpuLayers(input.gpuLayers),
    input.splitMode,
    Option.match(input.tensorSplit, { onNone: () => "none", onSome: (values) => values.join(",") }),
    input.kvCache.key,
    input.kvCache.value,
    canonicalFlashAttention(input.flashAttention),
    canonicalBatchSize(input.batchSize),
    canonicalMicroBatchSize(input.microBatchSize),
    input.mmap,
    input.mlock,
  ].join("\0")

  return {
    ...input,
    id: LlamaExecutionProfileId.make(createHash("sha256").update(canonical).digest("hex")),
  }
})

const appendContextSize = ContextSize.$match({
  ModelDefault: () => [] as string[],
  Tokens: ({ value }) => ["--ctx-size", String(value)],
})
const appendGpuLayers = GpuLayerSelection.$match({
  Fit: () => ["--fit", "on"],
  Exact: ({ layers }) => ["--n-gpu-layers", String(layers), "--fit", "off"],
})
const appendFlashAttention = FlashAttentionSelection.$match({
  RuntimeDefault: () => [] as string[],
  Enabled: () => ["--flash-attn", "on"],
  Disabled: () => ["--flash-attn", "off"],
})
const appendBatchSize = BatchSize.$match({
  RuntimeDefault: () => [] as string[],
  Exact: ({ value }) => ["--batch-size", String(value)],
})
const appendMicroBatchSize = MicroBatchSize.$match({
  RuntimeDefault: () => [] as string[],
  Exact: ({ value }) => ["--ubatch-size", String(value)],
})

export const renderExecutionProfileArguments = (profile: LlamaExecutionProfile): readonly string[] => {
  const arguments_ = [
    "--parallel", String(profile.parallelSlots),
    "--kv-unified",
    "--cont-batching",
    "--split-mode", profile.splitMode,
    "--cache-type-k", profile.kvCache.key,
    "--cache-type-v", profile.kvCache.value,
  ]
  arguments_.push(...appendContextSize(profile.contextSize))
  arguments_.push(...appendGpuLayers(profile.gpuLayers))
  Option.map(profile.tensorSplit, (values) => arguments_.push("--tensor-split", values.join(",")))
  arguments_.push(...appendFlashAttention(profile.flashAttention))
  arguments_.push(...appendBatchSize(profile.batchSize))
  arguments_.push(...appendMicroBatchSize(profile.microBatchSize))
  arguments_.push(profile.mmap ? "--mmap" : "--no-mmap")
  if (profile.mlock) arguments_.push("--mlock")
  return arguments_
}

const presetContextSize = ContextSize.$match({ ModelDefault: () => [] as string[], Tokens: ({ value }) => [`ctx-size = ${value}`] })
const presetGpuLayers = GpuLayerSelection.$match({ Fit: () => ["fit = on"], Exact: ({ layers }) => [`n-gpu-layers = ${layers}`, "fit = off"] })
const presetFlashAttention = FlashAttentionSelection.$match({ RuntimeDefault: () => [] as string[], Enabled: () => ["flash-attn = on"], Disabled: () => ["flash-attn = off"] })
const presetBatchSize = BatchSize.$match({ RuntimeDefault: () => [] as string[], Exact: ({ value }) => [`batch-size = ${value}`] })
const presetMicroBatchSize = MicroBatchSize.$match({ RuntimeDefault: () => [] as string[], Exact: ({ value }) => [`ubatch-size = ${value}`] })
const presetOutputLimit = OutputLimit.$match({ RuntimeDefault: () => [] as string[], Tokens: ({ value }) => [`n-predict = ${value}`] })

export const renderExecutionProfilePreset = (profile: LlamaExecutionProfile): readonly string[] => {
  const lines = [
    `parallel = ${profile.parallelSlots}`,
    "kv-unified = true",
    "cont-batching = true",
    "kv-offload = true",
    `split-mode = ${profile.splitMode}`,
    `cache-type-k = ${profile.kvCache.key}`,
    `cache-type-v = ${profile.kvCache.value}`,
    `mmap = ${profile.mmap}`,
    `mlock = ${profile.mlock}`,
  ]
  lines.push(...presetContextSize(profile.contextSize))
  lines.push(...presetGpuLayers(profile.gpuLayers))
  Option.map(profile.tensorSplit, (values) => lines.push(`tensor-split = ${values.join(",")}`))
  lines.push(...presetFlashAttention(profile.flashAttention))
  lines.push(...presetBatchSize(profile.batchSize))
  lines.push(...presetMicroBatchSize(profile.microBatchSize))
  lines.push(...presetOutputLimit(profile.outputLimit))
  return lines
}
