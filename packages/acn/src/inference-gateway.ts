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
export const LOCAL_CODEX_MODEL_PREFIX = "magnitude-local/"
export const MAX_ANTHROPIC_ROUTING_BODY_BYTES = 32 * 1024 * 1024
export const MAX_CODEX_ROUTING_BODY_BYTES = 128 * 1024 * 1024
export const CLAUDE_CODE_PROXY_PREFIX = "/inference/anthropic/proxies/claude-code"
export const CODEX_PROXY_PREFIX = "/inference/v1/proxies/codex"

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
  init?: RequestInit & { readonly decompress?: boolean },
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

export class CodexGateway extends Context.Tag("@magnitudedev/acn/CodexGateway")<
  CodexGateway,
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
      decompress: false,
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

const invalidAnthropic = (message: string, status = 400): RoutingBodyResult => ({
  _tag: "Invalid",
  response: anthropicGatewayError(
    message,
    status,
    status === 413 ? "request_too_large" : "invalid_request_error",
  ),
})

const readBoundedBody = async (
  source: Request,
  limit: number,
): Promise<Uint8Array | undefined> => {
  const declaredLength = source.headers.get("content-length")
  if (declaredLength !== null) {
    const length = Number(declaredLength)
    if (Number.isFinite(length) && length > limit) {
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
      if (length > limit) {
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

const classifyRoutingBytes = (
  bytes: Uint8Array,
  invalidResult: (message: string, status?: number) => RoutingBodyResult,
): RoutingBodyResult => {
  let text: string
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes)
  } catch {
    return invalidResult("Request body must be valid UTF-8 JSON")
  }
  const located = locateModelValue(text)
  if (located === undefined) {
    return invalidResult("Request body must contain exactly one top-level string model")
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

const classifyRoutingBody = async (source: Request): Promise<RoutingBodyResult> => {
  const bytes = await readBoundedBody(source, MAX_ANTHROPIC_ROUTING_BODY_BYTES)
  if (bytes === undefined) {
    return invalidAnthropic("Request body exceeds the 32 MB routing limit", 413)
  }
  return classifyRoutingBytes(bytes, invalidAnthropic)
}

const rewrittenLocalBody = (body: RoutingBody, canonicalModel: string): Uint8Array =>
  new TextEncoder().encode(
    `${body.source.slice(0, body.modelValueStart)}${JSON.stringify(canonicalModel)}${body.source.slice(body.modelValueEnd)}`,
  )

const localResponseItemId = (value: unknown): value is string =>
  typeof value === "string" && /^(?:msg|rs)_icn_/.test(value)

/**
 * Codex replays completed response items on later turns. OpenAI treats an
 * item carrying an ID as a reference to an item stored by OpenAI, so IDs
 * minted by ICN must not cross back into the upstream service. A local
 * reasoning item has no portable encrypted payload and is omitted; message
 * items remain inline conversation content with only their local ID removed.
 */
const upstreamCodexBody = (
  body: RoutingBody,
): { readonly bytes: Uint8Array; readonly changed: boolean } => {
  const value = JSON.parse(body.source) as Record<string, unknown>
  let changed = false
  if (typeof value.previous_response_id === "string" && value.previous_response_id.startsWith("resp_icn_")) {
    delete value.previous_response_id
    changed = true
  }
  if (Array.isArray(value.input)) {
    value.input = value.input.flatMap((entry) => {
      if (entry === null || typeof entry !== "object" || Array.isArray(entry)) return [entry]
      const item = entry as Record<string, unknown>
      if (!localResponseItemId(item.id)) return [entry]
      changed = true
      if (item.type === "reasoning") return []
      const inline = { ...item }
      delete inline.id
      return [inline]
    })
  }
  return changed
    ? { bytes: new TextEncoder().encode(JSON.stringify(value)), changed: true }
    : { bytes: body.bytes, changed: false }
}

const requestBody = (bytes: Uint8Array): ArrayBuffer => {
  const body = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(body).set(bytes)
  return body
}

type GatewayModel = {
  readonly type: "model"
  readonly id: string
  readonly display_name: string
  readonly description: string
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
    decompress: false,
  })
  if (!response.ok) return responseFromUpstream(response)
  const value = await Schema.decodeUnknownPromise(IcnSchemas.OpenAiModelsResponse)(
    await response.json(),
  )
  const models = value.data.map((entry): GatewayModel => {
    return {
      type: "model",
      id: `${LOCAL_ANTHROPIC_MODEL_PREFIX}${entry.id}`,
      display_name: entry.name,
      description: entry.description,
      created_at: new Date(entry.created * 1000).toISOString(),
    }
  })
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
      decompress: false,
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
      decompress: false,
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
  const targetPath = incoming.pathname.slice(CLAUDE_CODE_PROXY_PREFIX.length) || "/"
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

const openAiGatewayError = (
  message: string,
  status: number,
  code: string,
): Response => Response.json({
  error: {
    message,
    type: "invalid_request_error",
    param: null,
    code,
  },
}, { status })

const invalidCodex = (message: string, status = 400): RoutingBodyResult => ({
  _tag: "Invalid",
  response: openAiGatewayError(
    message,
    status,
    status === 413
      ? "request_too_large"
      : status === 415
        ? "unsupported_content_encoding"
        : "invalid_request",
  ),
})

const classifyCodexRoutingBody = async (source: Request): Promise<RoutingBodyResult> => {
  const encoding = source.headers.get("content-encoding")?.trim().toLowerCase()
  if (encoding !== undefined && encoding !== "" && encoding !== "identity" && encoding !== "zstd") {
    return invalidCodex(`Unsupported request Content-Encoding: ${encoding}`, 415)
  }
  const original = await readBoundedBody(source, MAX_CODEX_ROUTING_BODY_BYTES)
  if (original === undefined) {
    return invalidCodex("Request body exceeds the 128 MB routing limit", 413)
  }
  if (encoding !== "zstd") return classifyRoutingBytes(original, invalidCodex)
  let decoded: Uint8Array
  try {
    decoded = await Bun.zstdDecompress(original)
  } catch {
    return invalidCodex("Request body is not valid zstd content")
  }
  if (decoded.byteLength > MAX_CODEX_ROUTING_BODY_BYTES) {
    return invalidCodex("Decoded request body exceeds the 128 MB routing limit", 413)
  }
  const classified = classifyRoutingBytes(decoded, invalidCodex)
  return classified._tag === "Invalid"
    ? classified
    : {
        _tag: "Body",
        body: { ...classified.body, bytes: original },
      }
}

const fixedOpenAiOrigin = (headers: Headers): URL => headers.has("chatgpt-account-id")
  ? new URL("https://chatgpt.com/backend-api/codex/")
  : new URL("https://api.openai.com/v1/")

const fixedOpenAiUrl = (headers: Headers, targetPath: string, search: string): URL =>
  new URL(`${targetPath.replace(/^\//, "")}${search}`, fixedOpenAiOrigin(headers))

const websocketUrl = (url: URL): URL => {
  const target = new URL(url)
  target.protocol = target.protocol === "https:" ? "wss:" : "ws:"
  return target
}

const forwardedWebSocketHeaders = (source: Headers): Headers => {
  const headers = forwardedHeaders(source)
  for (const name of [...headers.keys()]) {
    if (name.startsWith("sec-websocket-")) headers.delete(name)
  }
  return headers
}

export type CodexWebSocketTarget =
  | {
      readonly _tag: "Target"
      readonly route: "local" | "upstream"
      readonly url: URL
      readonly headers: Headers
      readonly firstMessage: string | Uint8Array
    }
  | { readonly _tag: "Invalid"; readonly message: string }

/** Classifies one Responses WebSocket request frame and selects its fixed destination. */
export const codexWebSocketTarget = (
  firstMessage: string | Uint8Array,
  sourceHeaders: Headers,
  icn: InferenceProxyTarget,
): CodexWebSocketTarget => {
  const bytes = typeof firstMessage === "string"
    ? new TextEncoder().encode(firstMessage)
    : firstMessage
  const classified = classifyRoutingBytes(bytes, invalidCodex)
  if (classified._tag === "Invalid") {
    return { _tag: "Invalid", message: "First WebSocket frame must contain one top-level model" }
  }
  const { body } = classified
  const headers = forwardedWebSocketHeaders(sourceHeaders)
  if (!body.model.startsWith(LOCAL_CODEX_MODEL_PREFIX)) {
    const upstream = upstreamCodexBody(body)
    if (upstream.changed) {
      headers.delete("content-encoding")
      headers.delete("content-length")
    }
    return {
      _tag: "Target",
      route: "upstream",
      url: websocketUrl(fixedOpenAiUrl(headers, "/responses", "")),
      headers,
      firstMessage: upstream.changed ? upstream.bytes : firstMessage,
    }
  }
  const canonicalModel = body.model.slice(LOCAL_CODEX_MODEL_PREFIX.length)
  if (canonicalModel.length === 0) {
    return { _tag: "Invalid", message: "Local model alias is missing its canonical model ID" }
  }
  localAuthorization(headers, icn)
  return {
    _tag: "Target",
    route: "local",
    url: websocketUrl(new URL("/v1/responses", icn.origin)),
    headers,
    firstMessage: rewrittenLocalBody(body, canonicalModel),
  }
}

const proxyOpaqueCodexUpstream = async (
  source: Request,
  incoming: URL,
  targetPath: string,
  fetchTarget: InferenceFetch,
  signal: AbortSignal,
): Promise<Response> => {
  const headers = forwardedHeaders(source.headers)
  const response = await fetchTarget(fixedOpenAiUrl(headers, targetPath, incoming.search), {
    method: source.method,
    headers,
    body: source.body,
    signal,
    redirect: "manual",
    decompress: false,
  })
  return responseFromUpstream(response)
}

/**
 * Codex routing shim. Only POST /responses and its top-level model field are
 * understood; every OpenAI protocol field and every response byte is opaque.
 */
export const proxyCodexInferenceRequest = async (
  source: Request,
  icn: InferenceProxyTarget,
  fetchTarget: InferenceFetch = fetch,
  signal: AbortSignal = source.signal,
): Promise<Response> => {
  const incoming = new URL(source.url)
  const targetPath = incoming.pathname.slice(CODEX_PROXY_PREFIX.length) || "/"
  if (source.method !== "POST" || targetPath !== "/responses") {
    return proxyOpaqueCodexUpstream(source, incoming, targetPath, fetchTarget, signal)
  }

  const classified = await classifyCodexRoutingBody(source)
  if (classified._tag === "Invalid") return classified.response
  const { body } = classified
  const headers = forwardedHeaders(source.headers)
  if (!body.model.startsWith(LOCAL_CODEX_MODEL_PREFIX)) {
    const upstream = upstreamCodexBody(body)
    if (upstream.changed) {
      headers.delete("content-encoding")
      headers.delete("content-length")
    }
    const response = await fetchTarget(fixedOpenAiUrl(headers, targetPath, incoming.search), {
      method: source.method,
      headers,
      body: requestBody(upstream.bytes),
      signal,
      redirect: "manual",
      decompress: false,
    })
    return responseFromUpstream(response)
  }

  const canonicalModel = body.model.slice(LOCAL_CODEX_MODEL_PREFIX.length)
  if (canonicalModel.length === 0) {
    return openAiGatewayError(
      "Local model alias is missing its canonical model ID",
      400,
      "invalid_model",
    )
  }
  localAuthorization(headers, icn)
  headers.delete("content-encoding")
  headers.delete("content-length")
  const response = await fetchTarget(new URL("/v1/responses", icn.origin), {
    method: source.method,
    headers,
    body: requestBody(rewrittenLocalBody(body, canonicalModel)),
    signal,
    redirect: "manual",
    decompress: false,
  })
  return responseFromUpstream(response)
}

export const proxyLocalAnthropicInferenceRequest = async (
  source: Request,
  icn: InferenceProxyTarget,
  fetchTarget: InferenceFetch = fetch,
  signal: AbortSignal = source.signal,
): Promise<Response> => {
  const incoming = new URL(source.url)
  const targetPath = incoming.pathname.slice("/inference".length) || "/"
  if (targetPath !== "/anthropic" && !targetPath.startsWith("/anthropic/")) {
    return new Response("Not found", { status: 404 })
  }
  const headers = forwardedHeaders(source.headers)
  localAuthorization(headers, icn)
  const response = await fetchTarget(new URL(`${targetPath}${incoming.search}`, icn.origin), {
    method: source.method,
    headers,
    body: source.body,
    signal,
    redirect: "manual",
    decompress: false,
  })
  return responseFromUpstream(response)
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

export const makeCodexGateway = (
  icn: InferenceProxyTarget,
  fetchTarget: InferenceFetch = fetch,
): Context.Tag.Service<CodexGateway> => CodexGateway.of({
  route: (source) => Effect.tryPromise({
    try: (signal) => proxyCodexInferenceRequest(source, icn, fetchTarget, signal),
    catch: (cause) => new InferenceGatewayFailed({ cause }),
  }),
})
