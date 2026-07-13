import { Effect } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import type { LlamaCppModelsResponse, LlamaCppRawModel, LlamaCppModelMeta } from "./contract"

export interface LlamaCppDiscoveryConfig {
  readonly endpoint: string
  readonly auth?: (headers: Headers) => void
}

/**
 * Fetch the model list from a Llama.cpp server (GET /v1/models).
 * Returns raw model entries in OpenAI-compatible format.
 */
export function fetchModelList(
  config: LlamaCppDiscoveryConfig,
): Effect.Effect<readonly LlamaCppRawModel[], Error, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient

    const headers = new Headers()
    config.auth?.(headers)

    const headerRecord: Record<string, string> = {}
    headers.forEach((value, key) => {
      headerRecord[key] = value
    })

    const request = HttpClientRequest.get(`${config.endpoint}/v1/models`).pipe(
      HttpClientRequest.setHeaders(headerRecord),
    )

    const response = yield* client
      .execute(request)
      .pipe(
        Effect.mapError((cause) =>
          new Error(`Failed to connect to Llama.cpp server at ${config.endpoint}: ${String(cause)}`),
        ),
      )

    if (response.status < 200 || response.status >= 300) {
      const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
      return yield* Effect.fail(new Error(`Llama.cpp server returned HTTP ${response.status}: ${body}`))
    }

    const body = yield* response.json.pipe(
      Effect.mapError((cause) => new Error(`Failed to read model list response: ${cause}`)),
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
): Effect.Effect<{ readonly nCtx?: number } | null, never, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient

    const headers = new Headers()
    config.auth?.(headers)

    const headerRecord: Record<string, string> = {}
    headers.forEach((value, key) => {
      headerRecord[key] = value
    })

    const request = HttpClientRequest.get(`${config.endpoint}/props`).pipe(
      HttpClientRequest.setHeaders(headerRecord),
    )

    const response = yield* client.execute(request).pipe(
      Effect.catchAll(() => Effect.succeed(null)),
    )

    if (response === null) return null
    if (response.status < 200 || response.status >= 300) return null

    const body = yield* response.json.pipe(Effect.orElseSucceed(() => null))
    if (body === null) return null

    const props = body as { readonly default_generation_settings?: { readonly n_ctx?: number } }
    return {
      ...(props.default_generation_settings?.n_ctx !== undefined
        ? { nCtx: props.default_generation_settings.n_ctx }
        : {}),
    }
  })
}

/**
 * Derive a display name from GGUF metadata or model ID.
 */
export function deriveDisplayName(raw: LlamaCppRawModel): string {
  const meta = raw.meta
  if (meta?.general_name) return meta.general_name
  if (meta?.general_basename && meta?.general_version) {
    return `${meta.general_basename}-${meta.general_version}`
  }
  if (meta?.general_basename) return meta.general_basename
  // Fall back to the ID with path/extension stripped
  const id = raw.id
  const lastSlash = id.lastIndexOf("/")
  const stripped = lastSlash !== -1 ? id.slice(lastSlash + 1) : id
  // Remove .gguf extension and quantization suffix
  return stripped.replace(/\.gguf$/i, "").replace(/-[A-Z]\d+.*$/i, "")
}

/**
 * Extract context window from metadata, with fallbacks.
 */
export function deriveContextWindow(
  raw: LlamaCppRawModel,
  serverProps: { readonly nCtx?: number } | null,
  fallback = 4096,
): number {
  if (raw.meta?.n_ctx) return raw.meta.n_ctx
  if (raw.meta?.n_ctx_train) return raw.meta.n_ctx_train
  if (serverProps?.nCtx) return serverProps.nCtx
  return fallback
}

/**
 * Detect vision capability from metadata.
 */
export function detectVision(raw: LlamaCppRawModel): boolean {
  const arch = raw.meta?.general_architecture?.toLowerCase() ?? ""
  const name = (raw.meta?.general_name ?? raw.id).toLowerCase()
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
