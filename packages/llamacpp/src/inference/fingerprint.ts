import { Effect, pipe } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import type { DetectedServer, InstanceModelRef, InstanceModelStatus, ServerMode } from "./types"

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null
}

function parsePort(endpoint: string): number {
  const m = endpoint.match(/:(\d+)$/)
  return m ? Number(m[1]) : 0
}

function parseInstanceModel(raw: unknown): InstanceModelRef | null {
  if (!isRecord(raw)) return null
  const id = typeof raw.id === "string" ? raw.id : (typeof raw.model === "string" ? raw.model : null)
  if (!id) return null
  // Determine status — llama.cpp /v1/models doesn't expose loading state directly,
  // so default to "loaded" (it's in the list, so it's loaded or loading)
  const status: InstanceModelStatus = "loaded"
  // Path might be in the model field or a separate field
  const path = typeof raw.model === "string" ? raw.model : null
  return { id, status, loadedByUs: false, path }
}

/**
 * Fingerprint a server endpoint to verify it's a llama.cpp server.
 * Checks `/props` for llama.cpp-specific fields (build_info, default_generation_settings),
 * then `/v1/models` for loaded models.
 *
 * Returns `null` if the endpoint is not a confirmed llama.cpp server.
 */
export function fingerprintServer(
  endpoint: string,
): Effect.Effect<DetectedServer | null, never, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient

    // Must have /props with llama.cpp-specific fields
    const propsResponse = yield* pipe(
      client.execute(HttpClientRequest.get(`${endpoint}/props`)),
      Effect.catchAll(() => Effect.succeed(null)),
    )

    if (!propsResponse || propsResponse.status < 200 || propsResponse.status >= 300) return null

    const props = yield* pipe(
      propsResponse.json,
      Effect.orElseSucceed(() => null),
    )
    if (!isRecord(props)) return null

    // Require build_info — no other server has this
    if (typeof props.build_info !== "string") return null
    // Require default_generation_settings — llama.cpp-specific shape
    if (!isRecord(props.default_generation_settings)) return null

    // Router mode: /props has "role": "router"
    const mode: ServerMode = props.role === "router" ? "router" : "single-model"
    const buildInfo = props.build_info

    // Secondary check: /v1/models
    const modelsResponse = yield* pipe(
      client.execute(HttpClientRequest.get(`${endpoint}/v1/models`)),
      Effect.catchAll(() => Effect.succeed(null)),
    )

    if (modelsResponse && modelsResponse.status >= 200 && modelsResponse.status < 300) {
      const body = yield* pipe(
        modelsResponse.json,
        Effect.orElseSucceed(() => null),
      )
      if (isRecord(body) && Array.isArray(body.data)) {
        const models = body.data
          .map(parseInstanceModel)
          .filter((m): m is InstanceModelRef => m !== null)
        return { endpoint, port: parsePort(endpoint), mode, models, buildInfo }
      }
    }

    return { endpoint, port: parsePort(endpoint), mode, models: [], buildInfo }
  })
}
