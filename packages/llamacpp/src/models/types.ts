import { Schema } from "effect"

// ── Model metadata ──

/** Provenance of a discovered model — where it was found on disk. */
export const LocalModelSource = Schema.Union(
  Schema.TaggedStruct("hf-cache", {
    /** HuggingFace repo ID (e.g. `unsloth/gemma-4-E4B-it-GGUF`). */
    repoId: Schema.String,
    /** Git commit hash of the snapshot. */
    commit: Schema.String,
  }),
  Schema.TaggedStruct("user-dir", {
    /** Directory path the model was discovered in. */
    dir: Schema.String,
  }),
)
export type LocalModelSource = Schema.Schema.Type<typeof LocalModelSource>

/**
 * Information about a locally discovered GGUF model on disk.
 * Produced by the discovery scanner from GGUF metadata + filesystem info.
 */
export const LocalModelInfo = Schema.Struct({
  /** Unique ID: `repoId:filename` for HF models, or path-based hash for non-HF. */
  id: Schema.String,
  /** Human-readable display name from GGUF metadata or filename. */
  displayName: Schema.String,
  /** Absolute path to the first shard (or the single file). */
  filePath: Schema.String,
  /** All shard paths if the model is split across multiple files. */
  shardPaths: Schema.optional(Schema.Array(Schema.String)),
  /** Paired mmproj projector path, if any. */
  mmprojPath: Schema.optional(Schema.String),

  // ── Core metadata ──
  /** Model architecture (e.g. `llama`, `qwen2`, `gemma2`). */
  architecture: Schema.optional(Schema.String),
  /** Quantization label (e.g. `Q4_K_M`, `F16`). */
  quantization: Schema.optional(Schema.String),
  /** Trained context length from GGUF metadata. */
  contextLength: Schema.optional(Schema.Number),
  /** File size in bytes (first shard or total of all shards). */
  fileSizeBytes: Schema.Number,

  // ── Extended metadata ──
  /** Total parameter count (computed by `@huggingface/gguf`). */
  parameterCount: Schema.optional(Schema.Number),
  /** Hidden/embedding dimension. */
  hiddenSize: Schema.optional(Schema.Number),
  /** Number of transformer blocks/layers. */
  layerCount: Schema.optional(Schema.Number),
  /** Number of attention heads. */
  headCount: Schema.optional(Schema.Number),
  /** Vocabulary size. */
  vocabSize: Schema.optional(Schema.Number),
  /** Feed-forward network hidden dimension. */
  feedForwardLength: Schema.optional(Schema.Number),
  /** Number of experts (MoE models). */
  expertCount: Schema.optional(Schema.Number),
  /** Number of experts used per token (MoE models). */
  expertUsedCount: Schema.optional(Schema.Number),

  // ── Tokenizer ──
  /** Tokenizer model (e.g. `llama`, `gpt2`). */
  tokenizerModel: Schema.optional(Schema.String),

  // ── Capabilities ──
  /** Whether the model has vision/multimodal support (mmproj paired). */
  vision: Schema.Boolean,
  /** Whether the model has audio support. */
  audio: Schema.Boolean,
  /** Whether the model is a mixture-of-experts model. */
  moe: Schema.Boolean,

  // ── Provenance ──
  /** Where the model was discovered. */
  source: LocalModelSource,
  /** HF repo ID if from HF cache, otherwise absent. */
  repoId: Schema.optional(Schema.String),
  /** HF commit hash if from HF cache, otherwise absent. */
  commit: Schema.optional(Schema.String),
  /** Base model names from GGUF metadata (e.g. for merged models). */
  baseModelNames: Schema.optional(Schema.Array(Schema.String)),
})
export type LocalModelInfo = Schema.Schema.Type<typeof LocalModelInfo>

/** Options for the model discovery scan. */
export const DiscoverOptions = Schema.Struct({
  /** Additional user-configured directories to scan. */
  extraDirs: Schema.optional(Schema.Array(Schema.String)),
  /** Force re-scan (ignore metadata stat cache). */
  forceRefresh: Schema.optional(Schema.Boolean),
})
export type DiscoverOptions = Schema.Schema.Type<typeof DiscoverOptions>

/**
 * Structured GGUF metadata extracted from the library's flat KV record.
 * Architecture-specific keys are resolved using `general.architecture` as prefix.
 * Quantization is mapped from `general.file_type` via the `GGMLFileQuantizationType` enum.
 */
export const ExpandedGgufMetadata = Schema.Struct({
  // ── General metadata ──
  generalName: Schema.optional(Schema.String),
  generalBasename: Schema.optional(Schema.String),
  generalSizeLabel: Schema.optional(Schema.String),
  generalFinetune: Schema.optional(Schema.String),
  generalVersion: Schema.optional(Schema.String),

  // ── Architecture & quantization ──
  architecture: Schema.optional(Schema.String),
  quantization: Schema.optional(Schema.String),
  contextLength: Schema.optional(Schema.Number),
  hiddenSize: Schema.optional(Schema.Number),
  layerCount: Schema.optional(Schema.Number),
  headCount: Schema.optional(Schema.Number),
  vocabSize: Schema.optional(Schema.Number),
  expertCount: Schema.optional(Schema.Number),
  expertUsedCount: Schema.optional(Schema.Number),
  feedForwardLength: Schema.optional(Schema.Number),
  parameterCount: Schema.optional(Schema.Number),

  // ── Tokenizer ──
  tokenizerModel: Schema.optional(Schema.String),
  tokenizerPre: Schema.optional(Schema.String),
  chatTemplate: Schema.optional(Schema.String),
  chatTemplatePresent: Schema.Boolean,

  // ── Base model info ──
  baseModelNames: Schema.Array(Schema.String),
  baseModelRepositories: Schema.Array(Schema.String),
})
export type ExpandedGgufMetadata = Schema.Schema.Type<typeof ExpandedGgufMetadata>

/** A group of sharded GGUF files belonging to the same model. */
export const ShardGroup = Schema.Struct({
  /** Common prefix across all shards. */
  prefix: Schema.String,
  /** Total number of shards in the group. */
  total: Schema.Number,
  /** Shard file paths ordered by shard number. */
  shards: Schema.Array(Schema.String),
  /** Path to the first shard. */
  primaryPath: Schema.String,
})
export type ShardGroup = Schema.Schema.Type<typeof ShardGroup>

// ── Download types ──

/** Parameters for downloading a GGUF model from HuggingFace. */
export const DownloadModelParams = Schema.Struct({
  /** HuggingFace repo ID (e.g. `unsloth/gemma-4-E4B-it-GGUF`). */
  repo: Schema.String,
  /** GGUF filename within the repo (e.g. `gemma-4-E4B-it-Q4_K_M.gguf`). */
  file: Schema.String,
  /** Git revision/branch. Defaults to `main` if omitted. */
  revision: Schema.optional(Schema.String),
})
export type DownloadModelParams = Schema.Schema.Type<typeof DownloadModelParams>

/** Result of a completed model download. Emitted as the terminal stream event. */
export const DownloadModelResult = Schema.Struct({
  /** Local file path in the HF cache. */
  filePath: Schema.String,
  /** HuggingFace repo ID. */
  repoId: Schema.String,
  /** Git commit hash of the downloaded snapshot. */
  commit: Schema.String,
})
export type DownloadModelResult = Schema.Schema.Type<typeof DownloadModelResult>

/** Progress event emitted during a model download. */
export const DownloadProgress = Schema.Struct({
  /** Bytes downloaded so far. */
  downloadedBytes: Schema.Number,
  /** Total file size in bytes. */
  totalBytes: Schema.Number,
  /** Completion percentage (0-100). */
  percent: Schema.Number,
  /** Rolling average download speed in bytes/second. */
  bytesPerSecond: Schema.Number,
  /** Estimated time remaining in seconds. */
  etaSeconds: Schema.Number,
})
export type DownloadProgress = Schema.Schema.Type<typeof DownloadProgress>

/** Lifecycle status of a download tracked in the registry. */
export const DownloadStatus = Schema.Literal(
  "downloading",
  "paused",
  "completed",
  "failed",
)
export type DownloadStatus = Schema.Schema.Type<typeof DownloadStatus>

/** Observable state of a download, tracked in the download registry. */
export const DownloadState = Schema.Struct({
  /** Download ID: `${repo}/${file}`. */
  id: Schema.String,
  /** HuggingFace repo ID. */
  repo: Schema.String,
  /** GGUF filename. */
  file: Schema.String,
  /** Current lifecycle status. */
  status: DownloadStatus,
  /** Bytes downloaded so far. */
  downloadedBytes: Schema.Number,
  /** Total file size in bytes. */
  totalBytes: Schema.Number,
  /** Completion percentage (0-100). */
  percent: Schema.Number,
  /** Rolling average download speed in bytes/second. */
  bytesPerSecond: Schema.Number,
  /** Estimated time remaining in seconds. */
  etaSeconds: Schema.Number,
  /** Error message if `status === "failed"`. */
  error: Schema.optional(Schema.String),
})
export type DownloadState = Schema.Schema.Type<typeof DownloadState>

/** A GGUF file available in a HuggingFace repo. */
export const RepoGgufFile = Schema.Struct({
  /** Filename within the repo. */
  path: Schema.String,
  /** File size in bytes. */
  size: Schema.Number,
  /** Quantization label parsed from the filename (e.g. `Q4_K_M`). */
  quantization: Schema.optional(Schema.String),
})
export type RepoGgufFile = Schema.Schema.Type<typeof RepoGgufFile>

/** Events emitted by the download stream: progress updates and the terminal result. */
export const DownloadEvent = Schema.Union(DownloadProgress, DownloadModelResult)
export type DownloadEvent = Schema.Schema.Type<typeof DownloadEvent>
