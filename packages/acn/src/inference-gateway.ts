import * as IcnSchemas from "@magnitudedev/icn-protocol/schemas"
import { Context, Data, Effect, Schema } from "effect"
import { parseTree, type Node, type ParseError } from "jsonc-parser"

const HOP_BY_HOP_HEADERS = [
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
] as const

export const LOCAL_ANTHROPIC_MODEL_PREFIX = "anthropic-local/"
export const MAX_ANTHROPIC_ROUTING_BODY_BYTES = 32 * 1024 * 1024

let gatewayRequestId = 0

const anthropicGatewayError = (
  message: string,
  status: number,
  type: "invalid_request_error" | "not_found_error" | "request_too_large",
): Response => {
  const requestId = `req_acn_${++gatewayRequestId}`
  return Response.json({
    type: "error",
    error: { type, message },
    request_id: requestId,
  }, {
    status,
    headers: { "request-id": requestId },
  })
}

export interface InferenceProxyTarget {
  readonly origin: URL
  readonly clientOptions: {
    readonly headers?: Readonly<Record<string, string>>
  }
}

export type InferenceFetch = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>

export class InferenceGatewayFailed extends Data.TaggedError("InferenceGatewayFailed")<{
  readonly cause: unknown
}> {}

export class AnthropicGateway extends Context.Tag("@magnitudedev/acn/AnthropicGateway")<
  AnthropicGateway,
  {
    readonly route: (
      source: Request,
    ) => Effect.Effect<Response, InferenceGatewayFailed>
  }
>() {}

const forwardedHeaders = (source: Headers): Headers => {
  const headers = new Headers(source)
  headers.delete("host")
  headers.delete("x-magnitude-acn-id")
  headers.delete("magnitude-gateway-model")
  for (const header of HOP_BY_HOP_HEADERS) headers.delete(header)
  return headers
}

const responseFromUpstream = (response: Response): Response => {
  const headers = new Headers(response.headers)
  for (const header of HOP_BY_HOP_HEADERS) headers.delete(header)
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  })
}

const localAuthorization = (
  headers: Headers,
  icn: InferenceProxyTarget,
): void => {
  headers.delete("authorization")
  headers.delete("x-api-key")
  const authorization = icn.clientOptions.headers?.authorization
  if (authorization !== undefined) headers.set("authorization", authorization)
}

export const proxyOpenAiInferenceRequest = async (
  source: Request,
  icn: InferenceProxyTarget,
  fetchTarget: InferenceFetch = fetch,
  signal: AbortSignal = source.signal,
): Promise<Response> => {
  const incoming = new URL(source.url)
  const targetPath = incoming.pathname.slice("/inference".length) || "/"
  if (targetPath !== "/v1" && !targetPath.startsWith("/v1/")) {
    return new Response("Not found", { status: 404 })
  }
  const headers = forwardedHeaders(source.headers)
  localAuthorization(headers, icn)
  const response = await fetchTarget(
    new URL(`${targetPath}${incoming.search}`, icn.origin),
    {
      method: source.method,
      headers,
      body: source.body,
      signal,
      redirect: "manual",
    },
  )
  return responseFromUpstream(response)
}

type RoutingBody = {
  readonly bytes: Uint8Array
  readonly source: string
  readonly model: string
  readonly modelValueStart: number
  readonly modelValueEnd: number
}

type RoutingBodyResult =
  | { readonly _tag: "Body"; readonly body: RoutingBody }
  | { readonly _tag: "Invalid"; readonly response: Response }

const invalid = (message: string, status = 400): RoutingBodyResult => ({
  _tag: "Invalid",
  response: anthropicGatewayError(
    message,
    status,
    status === 413 ? "request_too_large" : "invalid_request_error",
  ),
})

const readBoundedBody = async (
  source: Request,
): Promise<Uint8Array | undefined> => {
  const declaredLength = source.headers.get("content-length")
  if (declaredLength !== null) {
    const length = Number(declaredLength)
    if (Number.isFinite(length) && length > MAX_ANTHROPIC_ROUTING_BODY_BYTES) {
      return undefined
    }
  }
  if (source.body === null) return new Uint8Array()
  const reader = source.body.getReader()
  const chunks: Uint8Array[] = []
  let length = 0
  try {
    while (true) {
      const next = await reader.read()
      if (next.done) break
      length += next.value.byteLength
      if (length > MAX_ANTHROPIC_ROUTING_BODY_BYTES) {
        await reader.cancel("routing body exceeded limit")
        return undefined
      }
      chunks.push(next.value)
    }
  } finally {
    reader.releaseLock()
  }
  const bytes = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return bytes
}

const propertyName = (node: Node): unknown => node.children?.[0]?.value

const locateModelValue = (
  source: string,
): { readonly model: string; readonly start: number; readonly end: number } | undefined => {
  const errors: ParseError[] = []
  const root = parseTree(source, errors, {
    allowTrailingComma: false,
    disallowComments: true,
  })
  if (errors.length > 0 || root?.type !== "object") return undefined
  const properties = root.children?.filter((node) => propertyName(node) === "model") ?? []
  if (properties.length !== 1) return undefined
  const value = properties[0]?.children?.[1]
  if (value?.type !== "string" || typeof value.value !== "string") return undefined
  return {
    model: value.value,
    start: value.offset,
    end: value.offset + value.length,
  }
}

const classifyRoutingBody = async (source: Request): Promise<RoutingBodyResult> => {
  const bytes = await readBoundedBody(source)
  if (bytes === undefined) {
    return invalid("Request body exceeds the 32 MB routing limit", 413)
  }
  let text: string
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes)
  } catch {
    return invalid("Request body must be valid UTF-8 JSON")
  }
  const located = locateModelValue(text)
  if (located === undefined) {
    return invalid("Request body must contain exactly one top-level string model")
  }
  return {
    _tag: "Body",
    body: {
      bytes,
      source: text,
      model: located.model,
      modelValueStart: located.start,
      modelValueEnd: located.end,
    },
  }
}

const rewrittenLocalBody = (body: RoutingBody, canonicalModel: string): Uint8Array =>
  new TextEncoder().encode(
    `${body.source.slice(0, body.modelValueStart)}${JSON.stringify(canonicalModel)}${body.source.slice(body.modelValueEnd)}`,
  )

const requestBody = (bytes: Uint8Array): ArrayBuffer => {
  const body = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(body).set(bytes)
  return body
}

type GatewayModel = {
  readonly type: "model"
  readonly id: string
  readonly display_name: string
  readonly created_at: string
}

const paginationError = (message: string): Response =>
  anthropicGatewayError(message, 400, "invalid_request_error")

const paginatedLocalModels = (
  models: ReadonlyArray<GatewayModel>,
  incoming: URL,
): Response => {
  const limitValue = incoming.searchParams.get("limit")
  const limit = limitValue === null ? 20 : Number(limitValue)
  if (!Number.isInteger(limit) || limit < 1 || limit > 1000) {
    return paginationError("limit must be an integer from 1 through 1000")
  }
  const before = incoming.searchParams.get("before_id")
  const after = incoming.searchParams.get("after_id")
  if (before !== null && after !== null) {
    return paginationError("before_id and after_id cannot be combined")
  }
  let start = 0
  let end = models.length
  if (after !== null) {
    const cursor = models.findIndex((model) => model.id === after)
    if (cursor < 0) return paginationError("after_id does not identify a model")
    start = cursor + 1
  }
  if (before !== null) {
    const cursor = models.findIndex((model) => model.id === before)
    if (cursor < 0) return paginationError("before_id does not identify a model")
    end = cursor
    start = Math.max(0, end - limit)
  }
  const available = models.slice(start, end)
  const data = before === null ? available.slice(0, limit) : available
  const hasMore = before === null ? available.length > data.length : start > 0
  return Response.json({
    data,
    has_more: hasMore,
    first_id: data.at(0)?.id ?? null,
    last_id: data.at(-1)?.id ?? null,
  })
}

const localModels = async (
  source: Request,
  incoming: URL,
  icn: InferenceProxyTarget,
  fetchTarget: InferenceFetch,
  signal: AbortSignal,
): Promise<Response> => {
  const headers = forwardedHeaders(source.headers)
  localAuthorization(headers, icn)
  const response = await fetchTarget(new URL("/v1/models", icn.origin), {
    method: "GET",
    headers,
    signal,
    redirect: "manual",
  })
  if (!response.ok) return responseFromUpstream(response)
  const value = await Schema.decodeUnknownPromise(IcnSchemas.OpenAiModelsResponse)(
    await response.json(),
  )
  const models = value.data.map((entry): GatewayModel => ({
        type: "model",
        id: `${LOCAL_ANTHROPIC_MODEL_PREFIX}${entry.id}`,
        display_name: entry.id,
        created_at: new Date(entry.created * 1000).toISOString(),
      }))
    .sort((left, right) => left.id.localeCompare(right.id))
  return paginatedLocalModels(models, incoming)
}

const proxyClassifiedRequest = async (
  source: Request,
  incoming: URL,
  targetPath: string,
  icn: InferenceProxyTarget,
  fetchTarget: InferenceFetch,
  signal: AbortSignal,
  anthropicOrigin: URL,
  localPath: string | undefined,
): Promise<Response> => {
  const classified = await classifyRoutingBody(source)
  if (classified._tag === "Invalid") return classified.response
  const { body } = classified
  const headers = forwardedHeaders(source.headers)
  if (body.model.startsWith(LOCAL_ANTHROPIC_MODEL_PREFIX)) {
    const canonicalModel = body.model.slice(LOCAL_ANTHROPIC_MODEL_PREFIX.length)
    if (canonicalModel.length === 0) {
      return anthropicGatewayError(
        "Local model alias is missing its canonical model ID",
        400,
        "invalid_request_error",
      )
    }
    if (localPath === undefined) {
      return anthropicGatewayError(
        "This operation is not available for local models",
        404,
        "not_found_error",
      )
    }
    localAuthorization(headers, icn)
    headers.set("magnitude-gateway-model", body.model)
    headers.delete("content-length")
    const response = await fetchTarget(new URL(localPath, icn.origin), {
      method: source.method,
      headers,
      body: requestBody(rewrittenLocalBody(body, canonicalModel)),
      signal,
      redirect: "manual",
    })
    return responseFromUpstream(response)
  }
  const response = await fetchTarget(
    new URL(`${targetPath}${incoming.search}`, anthropicOrigin),
    {
      method: source.method,
      headers,
      body: requestBody(body.bytes),
      signal,
      redirect: "manual",
    },
  )
  return responseFromUpstream(response)
}

/**
 * Dumb Anthropic routing shim. It understands only paths and the top-level
 * model discriminator; Anthropic request and streaming semantics remain opaque.
 */
export const proxyAnthropicInferenceRequest = async (
  source: Request,
  icn: InferenceProxyTarget,
  fetchTarget: InferenceFetch = fetch,
  signal: AbortSignal = source.signal,
  anthropicOrigin = new URL("https://api.anthropic.com"),
): Promise<Response> => {
  const incoming = new URL(source.url)
  const targetPath = incoming.pathname.slice("/inference/anthropic".length) || "/"
  if (source.method === "HEAD" && targetPath === "/api/hello") {
    return new Response(null, { status: 204 })
  }
  if (source.method === "GET" && targetPath === "/v1/models") {
    return localModels(source, incoming, icn, fetchTarget, signal)
  }
  if (source.method === "POST" && targetPath === "/v1/messages") {
    return proxyClassifiedRequest(
      source,
      incoming,
      targetPath,
      icn,
      fetchTarget,
      signal,
      anthropicOrigin,
      "/anthropic/v1/messages",
    )
  }
  if (source.method === "POST" && targetPath === "/v1/messages/count_tokens") {
    return proxyClassifiedRequest(
      source,
      incoming,
      targetPath,
      icn,
      fetchTarget,
      signal,
      anthropicOrigin,
      "/anthropic/v1/messages/count_tokens",
    )
  }
  return anthropicGatewayError("Not found", 404, "not_found_error")
}

export const makeAnthropicGateway = (
  icn: InferenceProxyTarget,
  fetchTarget: InferenceFetch = fetch,
  anthropicOrigin = new URL("https://api.anthropic.com"),
): Context.Tag.Service<AnthropicGateway> => AnthropicGateway.of({
  route: (source) => Effect.tryPromise({
    try: (signal) => proxyAnthropicInferenceRequest(
      source,
      icn,
      fetchTarget,
      signal,
      anthropicOrigin,
    ),
    catch: (cause) => new InferenceGatewayFailed({ cause }),
  }),
})
