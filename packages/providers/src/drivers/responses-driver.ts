import { Effect, Stream } from 'effect'
import { normalizeQuotesInString } from '../util/output-normalization'
import { buildUsage } from '../usage'
import type { CallUsage } from '../state/provider-state'
import type { DriverRequest, ExecutableDriver, StreamResult, CompleteResult } from './types'
import type { AuthInfo } from '../types'
import type { Model } from '../model/model'
import { CollectorData } from './types'
import { getCodexReasoningEffort } from '../reasoning-effort'
import { classifyHttpError, classifyUnknownError } from '../errors/classify-error'
import { buildClientRegistry } from '../client-registry-builder'
import { bamlParse, bamlStreamRequest } from './baml-dispatch'
import { COPILOT_HEADERS } from '../auth/copilot-oauth'
import { ModelConnection as ModelConnectionCtor } from '../model/model-connection'
import type { InferenceConfig } from '../model/inference-config'
import type { ModelError } from '../errors/model-error'


type CodexVariant = 'openai-codex' | 'copilot-codex'

function getCodexVariant(model: Model, auth: AuthInfo | null): CodexVariant | null {
  if (model.providerId === 'openai' && auth?.type === 'oauth') return 'openai-codex'
  if (model.providerId === 'github-copilot' && model.id.includes('codex')) return 'copilot-codex'
  return null
}

function getResponsesEndpoint(model: Model, auth: AuthInfo | null): string {
  const variant = getCodexVariant(model, auth)
  if (variant === 'openai-codex') return 'https://chatgpt.com/backend-api/codex/responses'
  if (variant === 'copilot-codex') return 'https://api.githubcopilot.com/v1/responses'
  return 'https://api.openai.com/v1/responses'
}

function getHeaders(model: Model, auth: AuthInfo | null): Record<string, string> {
  const variant = getCodexVariant(model, auth)
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }

  if (auth?.type === 'oauth') {
    headers.Authorization = `Bearer ${auth.accessToken}`
    if (variant === 'openai-codex' && auth.accountId) headers['ChatGPT-Account-Id'] = auth.accountId
    if (variant === 'copilot-codex') {
      headers['Openai-Intent'] = 'conversation-edits'
      headers['x-initiator'] = 'user'
      Object.assign(headers, COPILOT_HEADERS)
    }
    return headers
  }

  if (auth?.type === 'api') {
    headers.Authorization = `Bearer ${auth.key}`
    return headers
  }

  const envKey = process.env.OPENAI_API_KEY
  if (envKey) headers.Authorization = `Bearer ${envKey}`
  return headers
}

function transformForResponsesApi(
  body: Record<string, unknown>,
  stream: boolean,
  maxOutputTokens?: number,
): Record<string, unknown> {
  const inputValue = body.input ?? body.messages
  const messages = Array.isArray(inputValue) ? inputValue : []
  const systemParts: string[] = []
  const nonSystemMessages: unknown[] = []

  for (const msg of messages) {
    if (typeof msg === 'object' && msg !== null && 'role' in msg && (msg as { role: unknown }).role === 'system') {
      const content = (msg as { content?: unknown }).content
      const text = typeof content === 'string'
        ? content
        : Array.isArray(content)
          ? content.map((p) => (typeof p === 'object' && p !== null && 'text' in p ? ((p as { text?: unknown }).text ?? '') : '')).join('')
          : ''
      systemParts.push(text)
    } else {
      nonSystemMessages.push(msg)
    }
  }

  const result: Record<string, unknown> = {
    model: body.model,
    instructions: systemParts.join('\n\n'),
    input: nonSystemMessages,
    stream,
    store: false,
    ...(maxOutputTokens ? { max_output_tokens: maxOutputTokens } : {}),
    text: { verbosity: 'low' },
  }

  return result
}

function streamViaResponsesApi(req: DriverRequest): StreamResult {
  if (req.connection._tag !== 'Responses') {
    throw new Error('Invalid connection type for ResponsesDriver')
  }
  const connection = req.connection

  let inputTokens: number | null = null
  let outputTokens: number | null = null
  let rawResponseBody: unknown = null
  let rawRequestBody: unknown = null
  let sseEvents: unknown[] = []

  async function* generate(): AsyncGenerator<string> {
    const clientRegistry = buildClientRegistry(
      req.model.providerId,
      req.model.id,
      connection.auth,
      req.providerOptions,
      req.inference.stopSequences ? [...req.inference.stopSequences] : undefined,
    )
    const bamlReq = await bamlStreamRequest(req.functionName, req.args, { clientRegistry })
    const maxOutputTokens = getCodexVariant(req.model, connection.auth) !== null ? undefined : req.model.maxOutputTokens ?? undefined
    rawRequestBody = bamlReq.body.json()
    const transformed = transformForResponsesApi(
      rawRequestBody as Record<string, unknown>,
      true,
      maxOutputTokens,
    )

    // Apply reasoning effort if model supports it (driver-internal, provider-specific)
    const reasoningEffort = getCodexReasoningEffort(req.model.id)
    if (reasoningEffort && !transformed.reasoning) {
      transformed.reasoning = { effort: reasoningEffort }
    }

    const response = await fetch(connection.endpoint, {
      method: 'POST',
      headers: connection.headers,
      body: JSON.stringify(transformed),
      signal: req.signal,
    })

    if (!response.ok) {
      const text = await response.text()
      throw classifyHttpError(response.status, text)
    }

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6)
        if (data === '[DONE]') return

        try {
          const event = JSON.parse(data)
          sseEvents.push(event)
          if (event.type === 'response.output_text.delta') {
            yield normalizeQuotesInString(event.delta ?? '')
          } else if (event.type === 'response.completed') {
            rawResponseBody = event.response ?? null
            const usage = event.response?.usage
            if (typeof usage?.input_tokens === 'number') inputTokens = usage.input_tokens
            if (typeof usage?.output_tokens === 'number') outputTokens = usage.output_tokens
          }
        } catch {}
      }
    }
  }

  return {
    stream: Stream.fromAsyncIterable(generate(), (e) => classifyUnknownError(e)),
    getUsage(): CallUsage {
      const auth = req.connection._tag === 'Responses' ? req.connection.auth : null
      return buildUsage(req.model, auth?.type ?? null, inputTokens, outputTokens, null, null)
    },
    getCollectorData() {
      return CollectorData.Responses({ rawRequestBody, rawResponseBody, sseEvents })
    },
  }
}

export const ResponsesDriver: ExecutableDriver = {
  id: 'openai-responses',
  connect(model: Model, auth: AuthInfo | null, _inference: InferenceConfig): Effect.Effect<ReturnType<typeof ModelConnectionCtor.Responses>, ModelError> {
    return Effect.succeed(ModelConnectionCtor.Responses({
      auth,
      endpoint: getResponsesEndpoint(model, auth),
      headers: getHeaders(model, auth),
    }))
  },
  stream(req: DriverRequest) {
    return Effect.try({
      try: () => streamViaResponsesApi(req),
      catch: (error) => classifyUnknownError(error),
    })
  },
  complete<T = unknown>(req: DriverRequest) {
    return Effect.flatMap(
      Effect.try({
        try: () => streamViaResponsesApi(req),
        catch: (error) => classifyUnknownError(error),
      }),
      ({ stream, getUsage, getCollectorData }) =>
        Effect.flatMap(
          Stream.runFold(stream, '', (acc, chunk) => acc + chunk),
          (output) =>
            Effect.flatMap(
              Effect.try({
                try: () => bamlParse(req.functionName, output),
                catch: (error) => classifyUnknownError(error),
              }),
              (result) =>
                Effect.succeed({
                  result: result as T,
                  usage: getUsage(),
                  collectorData: getCollectorData(),
                } satisfies CompleteResult<T>),
            ),
        ),
    )
  },
}