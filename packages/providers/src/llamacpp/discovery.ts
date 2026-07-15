import { Data, Effect, Option, Secret } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import { makeLlamaCppEndpointClient, type LlamaCppConnection } from "@magnitudedev/llamacpp/client"
import type {
  LlamaCppRawModel,
  ServerProps,
  ServerStatus,
} from "./contract"

export interface LlamaCppDiscoveryConfig {
  readonly endpoint: string
  readonly auth?: (headers: Headers) => void
}

export interface CheckServerHealthOptions {
  readonly timeoutMs?: number
}

export class LlamaCppDiscoveryError extends Data.TaggedError("LlamaCppDiscoveryError")<{
  readonly message: string
  readonly cause?: unknown
}> {}

function connectionFor(config: LlamaCppDiscoveryConfig): LlamaCppConnection {
  const headers = new Headers()
  config.auth?.(headers)
  const authorization = headers.get("authorization")
  const token = authorization?.match(/^Bearer\s+(.+)$/i)?.[1]
  return {
    baseUrl: config.endpoint,
    apiKey: token ? Option.some(Secret.fromString(token)) : Option.none(),
  }
}

export function checkServerHealth(
  endpoint: string,
  options?: CheckServerHealthOptions,
): Effect.Effect<ServerStatus, never, HttpClient.HttpClient> {
  const timeoutMs = options?.timeoutMs ?? 2_000

  return makeLlamaCppEndpointClient({ baseUrl: endpoint, apiKey: Option.none() }).health.pipe(
    Effect.timeout(`${timeoutMs} millis`),
    Effect.orElseSucceed(() => ({ _tag: "Unavailable", message: "Connection timed out" } as const)),
    Effect.map((health): ServerStatus => {
      switch (health._tag) {
        case "Ready": return { status: "ready", endpoint }
        case "Loading": return { status: "loading", endpoint }
        case "Unavailable": return health.message.startsWith("HTTP ")
          ? { status: "error", endpoint, message: health.message }
          : { status: "not_found", endpoint }
      }
    }),
  )
}

/**
 * Fetch the model list from a Llama.cpp server (GET /v1/models).
 * Returns raw model entries in OpenAI-compatible format.
 */
export function fetchModelList(
  config: LlamaCppDiscoveryConfig,
): Effect.Effect<readonly LlamaCppRawModel[], LlamaCppDiscoveryError, HttpClient.HttpClient> {
  return makeLlamaCppEndpointClient(connectionFor(config)).models.pipe(
    Effect.map((models): readonly LlamaCppRawModel[] => models),
    Effect.mapError((cause) => new LlamaCppDiscoveryError({ message: cause.reason, cause })),
  )
}

/**
 * Probe server props for context window info (GET /props).
 * Falls back gracefully if the endpoint doesn't exist.
 */
export function fetchServerProps(
  config: LlamaCppDiscoveryConfig,
): Effect.Effect<ServerProps | null, never, HttpClient.HttpClient> {
  return makeLlamaCppEndpointClient(connectionFor(config)).props.pipe(
    Effect.map((props): ServerProps => ({
      ...(props.default_generation_settings?.n_ctx !== undefined
        ? { nCtx: props.default_generation_settings.n_ctx }
        : {}),
      ...(props.model_alias !== undefined ? { modelAlias: props.model_alias } : {}),
      ...(props.model_ftype !== undefined ? { modelFtype: props.model_ftype } : {}),
      ...(props.model_path !== undefined ? { modelPath: props.model_path } : {}),
      ...(props.chat_template !== undefined ? { chatTemplate: props.chat_template } : {}),
      ...(props.modalities
        ? {
            modalities: {
              ...(props.modalities.vision === undefined ? {} : { vision: props.modalities.vision }),
              ...(props.modalities.audio === undefined ? {} : { audio: props.modalities.audio }),
            },
          }
        : {}),
    })),
    Effect.orElseSucceed(() => null),
  )
}

const MODEL_ARTIFACT_EXTENSION = /\.(?:gguf|ggml|bin)$/i
const MODEL_SHARD_SUFFIX = /[-.]?\d{5}-of-\d{5}$/i
const GENERIC_QUANTIZATION_SUFFIX = /^(.*?)[._:-]((?:UD[-_])?(?:(?:I?Q\d+(?:_[A-Z0-9]+)+)|(?:TQ\d+_\d+)|(?:MXFP\d+)|(?:BF16|F16|F32|F64|I8|I16|I32|I64)))$/i

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function metadataValue(
  meta: LlamaCppRawModel["meta"],
  standardKey: keyof NonNullable<LlamaCppRawModel["meta"]>,
  compatibilityKey: keyof NonNullable<LlamaCppRawModel["meta"]>,
): string | undefined {
  const standard = meta?.[standardKey]
  if (typeof standard === "string" && standard.trim()) return standard.trim()
  const compatibility = meta?.[compatibilityKey]
  return typeof compatibility === "string" && compatibility.trim()
    ? compatibility.trim()
    : undefined
}

export function deriveMetadataName(raw: LlamaCppRawModel): string | undefined {
  const direct = metadataValue(raw.meta, "general.name", "general_name")
  if (direct) return direct

  const basename = metadataValue(raw.meta, "general.basename", "general_basename")
  if (!basename) return undefined

  const parts = [basename]
  for (const value of [
    metadataValue(raw.meta, "general.size_label", "general_size_label"),
    metadataValue(raw.meta, "general.finetune", "general_finetune"),
    metadataValue(raw.meta, "general.version", "general_version"),
  ]) {
    if (value && !parts.some((part) => part.toLowerCase().includes(value.toLowerCase()))) {
      parts.push(value)
    }
  }
  return parts.join("-")
}

export function deriveModelArchitecture(raw: LlamaCppRawModel): string | undefined {
  return metadataValue(raw.meta, "general.architecture", "general_architecture")
}

export function deriveTokenizerModel(raw: LlamaCppRawModel): string | undefined {
  return metadataValue(raw.meta, "tokenizer.ggml.model", "tokenizer_ggml_model")
}

export function deriveTokenizerPre(raw: LlamaCppRawModel): string | undefined {
  return metadataValue(raw.meta, "tokenizer.ggml.pre", "tokenizer_ggml_pre")
}

const GGUF_PATH = /\.gguf$/i

function modelPathFromArgs(args: readonly string[] | undefined): string | undefined {
  if (!args) return undefined

  for (let index = 0; index < args.length; index++) {
    const arg = args[index]
    if (arg === undefined) continue
    if (arg === "-m" || arg === "--model") {
      const value = args[index + 1]?.trim()
      if (value && GGUF_PATH.test(value)) return value
    }
    if (arg.startsWith("--model=")) {
      const value = arg.slice("--model=".length).trim()
      if (value && GGUF_PATH.test(value)) return value
    }
    if (arg.startsWith("-m=")) {
      const value = arg.slice("-m=".length).trim()
      if (value && GGUF_PATH.test(value)) return value
    }
  }
  return undefined
}

/** Find a server-reported path that may refer to a locally accessible GGUF. */
export function deriveSourceModelPath(
  raw: LlamaCppRawModel,
  serverProps: ServerProps | null,
  modelCount: number,
): string | undefined {
  const candidates = [
    raw.path,
    modelCount === 1 ? serverProps?.modelPath : undefined,
    modelPathFromArgs(raw.status?.args),
    raw.id,
  ]
  return candidates.find((candidate) => {
    const value = candidate?.trim()
    return value !== undefined && value.toLowerCase() !== "none" && GGUF_PATH.test(value)
  })?.trim()
}

interface ParsedModelIdentifier {
  readonly name: string
  readonly quantization?: string
}

function splitQuantizationSuffix(name: string, ftype?: string): ParsedModelIdentifier {
  const normalizedFtype = ftype?.trim().split(/\s+-\s+/, 1)[0]
  if (normalizedFtype && /^[A-Z0-9_]+$/i.test(normalizedFtype)) {
    const exactFtypeSuffix = new RegExp(
      `^(.*?)[._:-]((?:UD[-_])?${escapeRegExp(normalizedFtype)}(?:_[A-Z0-9]+)*)$`,
      "i",
    )
    const exact = name.match(exactFtypeSuffix)
    if (exact?.[1] && exact[2]) {
      return { name: exact[1], quantization: exact[2] }
    }
  }

  const generic = name.match(GENERIC_QUANTIZATION_SUFFIX)
  return generic?.[1] && generic[2]
    ? { name: generic[1], quantization: generic[2] }
    : { name }
}

function parseModelIdentifier(id: string, ftype?: string): ParsedModelIdentifier {
  const trimmed = id.trim()
  const lastPathSeparator = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"))
  const leaf = (lastPathSeparator === -1 ? trimmed : trimmed.slice(lastPathSeparator + 1))
    .replace(MODEL_ARTIFACT_EXTENSION, "")
    .replace(MODEL_SHARD_SUFFIX, "")

  const parsed = splitQuantizationSuffix(leaf, ftype)
  const name = parsed.name
    .replace(/[-_.]GGUF$/i, "")
    .replace(/[-_.]GGML$/i, "")
    .replace(/[-_.]+$/, "")

  return {
    name: name || leaf || trimmed,
    ...(parsed.quantization ? { quantization: parsed.quantization } : {}),
  }
}

/**
 * Derive a display name from GGUF metadata or model ID.
 */
export function deriveDisplayName(
  raw: LlamaCppRawModel,
  serverProps?: ServerProps | null,
  sourceModelPath?: string,
): string {
  const ftype = raw.meta?.ftype ?? serverProps?.modelFtype
  const rawAlias = raw.aliases?.find((alias) => alias.trim() && alias !== raw.id)
    ?? raw.aliases?.find((alias) => alias.trim())
  const serverAlias = serverProps?.modelAlias?.trim() || rawAlias
  const aliasParsed = serverAlias
    ? parseModelIdentifier(serverAlias, ftype)
    : undefined
  const pathParsed = sourceModelPath ? parseModelIdentifier(sourceModelPath, ftype) : undefined
  const idParsed = parseModelIdentifier(raw.id || "Unknown model", ftype)
  const quantization = pathParsed?.quantization
    ?? aliasParsed?.quantization
    ?? idParsed.quantization
  const name = deriveMetadataName(raw)
    ?? aliasParsed?.name
    ?? pathParsed?.name
    ?? idParsed.name
  return quantization && !name.toLowerCase().includes(quantization.toLowerCase())
    ? `${name} (${quantization})`
    : name
}

/**
 * Extract context window from metadata, with fallbacks.
 */
export function deriveContextWindow(
  raw: LlamaCppRawModel,
  serverProps: ServerProps | null,
  fallback = 4096,
): number {
  if (serverProps?.nCtx) return serverProps.nCtx
  if (raw.meta?.n_ctx) return raw.meta.n_ctx
  if (raw.meta?.n_ctx_train) return raw.meta.n_ctx_train
  return fallback
}

/**
 * Detect vision capability from metadata.
 */
export function detectVision(raw: LlamaCppRawModel, serverProps: ServerProps | null): boolean {
  if (serverProps?.modalities) {
    return serverProps.modalities.vision ?? false
  }

  if (raw.architecture?.input_modalities) {
    return raw.architecture.input_modalities.some((modality) =>
      modality.toLowerCase() === "image" || modality.toLowerCase() === "video",
    )
  }

  const arch = deriveModelArchitecture(raw)?.toLowerCase() ?? ""
  const name = (deriveMetadataName(raw) ?? raw.id).toLowerCase()
  // Vision-capable architectures and model name patterns
  return (
    arch.includes("gemma3") ||
    arch.includes("llama4") ||
    arch.includes("qwen2vl") ||
    arch.includes("qwen2-vl") ||
    name.includes("vision") ||
    name.includes("vl") ||
    name.includes("mmproj")
  )
}
