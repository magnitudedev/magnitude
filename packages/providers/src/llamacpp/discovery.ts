import { Data, Effect } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import type {
  LlamaCppModelsResponse,
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function headersFromAuth(auth?: (headers: Headers) => void): Record<string, string> {
  const headers = new Headers()
  auth?.(headers)

  const headerRecord: Record<string, string> = {}
  headers.forEach((value, key) => {
    headerRecord[key] = value
  })
  return headerRecord
}

export function checkServerHealth(
  endpoint: string,
  options?: CheckServerHealthOptions,
): Effect.Effect<ServerStatus, never, HttpClient.HttpClient> {
  const timeoutMs = options?.timeoutMs ?? 2_000

  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const request = HttpClientRequest.get(`${endpoint}/health`)
    const response = yield* client.execute(request).pipe(
      Effect.timeout(`${timeoutMs} millis`),
      Effect.catchAll(() => Effect.succeed(null)),
    )

    if (response === null) {
      return { status: "not_found", endpoint }
    }
    if (response.status === 503) {
      return { status: "loading", endpoint }
    }
    if (response.status === 200) {
      const body = yield* response.json.pipe(Effect.orElseSucceed(() => null))
      if (isRecord(body) && body.status === "ok") {
        return { status: "ready", endpoint }
      }
      return {
        status: "error",
        endpoint,
        message: "llama-server returned an invalid health response",
      }
    }

    const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
    return {
      status: "error",
      endpoint,
      message: body.trim() || `llama-server returned HTTP ${response.status}`,
    }
  })
}

/**
 * Fetch the model list from a Llama.cpp server (GET /v1/models).
 * Returns raw model entries in OpenAI-compatible format.
 */
export function fetchModelList(
  config: LlamaCppDiscoveryConfig,
): Effect.Effect<readonly LlamaCppRawModel[], LlamaCppDiscoveryError, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient

    const request = HttpClientRequest.get(`${config.endpoint}/v1/models`).pipe(
      HttpClientRequest.setHeaders(headersFromAuth(config.auth)),
    )

    const response = yield* client
      .execute(request)
      .pipe(
        Effect.mapError((cause) =>
          new LlamaCppDiscoveryError({
            message: `Failed to connect to Llama.cpp server at ${config.endpoint}`,
            cause,
          }),
        ),
      )

    if (response.status < 200 || response.status >= 300) {
      const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
      return yield* new LlamaCppDiscoveryError({
        message: `Llama.cpp server returned HTTP ${response.status}: ${body}`,
      })
    }

    const body = yield* response.json.pipe(
      Effect.mapError((cause) => new LlamaCppDiscoveryError({
        message: "Failed to read model list response",
        cause,
      })),
    )

    const parsed = body as LlamaCppModelsResponse
    return parsed.data
  })
}

/**
 * Probe server props for context window info (GET /props).
 * Falls back gracefully if the endpoint doesn't exist.
 */
export function fetchServerProps(
  config: LlamaCppDiscoveryConfig,
): Effect.Effect<ServerProps | null, never, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const headers = headersFromAuth(config.auth)
    const fetchPropsAt = (path: string) =>
      client.execute(
        HttpClientRequest.get(`${config.endpoint}${path}`).pipe(
          HttpClientRequest.setHeaders(headers),
        ),
      ).pipe(Effect.catchAll(() => Effect.succeed(null)))

    const primary = yield* fetchPropsAt("/props")
    const response = primary && primary.status >= 200 && primary.status < 300
      ? primary
      : yield* fetchPropsAt("/v1/props")

    if (response === null || response.status < 200 || response.status >= 300) return null

    const body = yield* response.json.pipe(Effect.orElseSucceed(() => null))
    if (!isRecord(body)) return null

    const generationSettings = isRecord(body.default_generation_settings)
      ? body.default_generation_settings
      : null
    const modalities = isRecord(body.modalities) ? body.modalities : null

    return {
      ...(typeof generationSettings?.n_ctx === "number"
        ? { nCtx: generationSettings.n_ctx }
        : {}),
      ...(typeof body.model_alias === "string" ? { modelAlias: body.model_alias } : {}),
      ...(typeof body.model_ftype === "string" ? { modelFtype: body.model_ftype } : {}),
      ...(typeof body.model_path === "string" ? { modelPath: body.model_path } : {}),
      ...(typeof body.chat_template === "string" ? { chatTemplate: body.chat_template } : {}),
      ...(modalities
        ? {
            modalities: {
              ...(typeof modalities.vision === "boolean" ? { vision: modalities.vision } : {}),
              ...(typeof modalities.audio === "boolean" ? { audio: modalities.audio } : {}),
            },
          }
        : {}),
    }
  })
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
    const arg = args[index]!
    if (arg === "-m" || arg === "--model") {
      const value = args[index + 1]?.trim()
      if (value && GGUF_PATH.test(value)) return value
    }
    if (arg.startsWith("--model=")) {
      const value = arg.slice("--model=".length).trim()
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
  const aliasParsed = serverProps?.modelAlias
    ? parseModelIdentifier(serverProps.modelAlias, ftype)
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
