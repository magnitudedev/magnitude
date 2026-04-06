import { Effect, Stream } from 'effect'
import { logger } from '@magnitudedev/logger'
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
import { normalizeResponsesUsage } from './usage-normalization'


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
  // Temporary verification aid: keep verbose usage diagnostics opt-in only.
  const usageDiagnosticsEnabled = process.env.MAGNITUDE_USAGE_DIAGNOSTICS === '1'

  let rawResponseBody: unknown = null
  let rawRequestBody: unknown = null
  const sseEvents: unknown[] = []
  let terminalEventType: string | null = null
  let terminalEventPayload: unknown = null
  const rawStreamTailLimit = 12
  const rawStreamTail: string[] = []
  let sawDoneSentinel = false
  let sawEof = false
  let sawAbort = false
  let streamError: string | null = null
  let streamEndReason: 'done-sentinel' | 'eof' | 'aborted' | 'error' | 'unknown' = 'unknown'
  const eventTypeCounts = new Map<string, number>()
  let parsedEventCount = 0
  let usageBearingEventCount = 0
  let responseIdSeen = false
  let responseId: string | null = null

  type UsageSource =
    | 'response.completed.response.usage'
    | 'response.completed.usage'
    | 'response.other'
    | 'raw-response-body.usage'
    | 'fallback-retrieve.response.usage'
    | 'fallback-retrieve.usage'
    | 'none'
  type UsagePath =
    | 'event.response.usage'
    | 'event.usage'
    | 'response.completed.response.usage'
    | 'response.completed.usage'
    | 'rawResponseBody.usage'
    | 'fallbackRetrieve.response.usage'
    | 'fallbackRetrieve.usage'

  type UsageSelection = {
    readonly usageSource: UsageSource
    readonly usagePath: UsagePath | 'none'
    readonly eventType: string | null
    readonly rawUsage: unknown | null
    readonly inputTokens: number | null
    readonly outputTokens: number | null
    readonly cacheReadTokens: number | null
    readonly cacheWriteTokens: number | null
  }

  let completedResponseUsage: UsageSelection | null = null
  let completedTopLevelUsage: UsageSelection | null = null
  let latestOtherUsage: UsageSelection | null = null
  let fallbackRetrievedUsage: UsageSelection | null = null
  let fallbackRetrieveUsed = false
  let fallbackRetrieveSucceeded = false
  let fallbackRetrieveUsageFound = false
  let fallbackRetrieveUsagePath: UsagePath | null = null
  const usageRejectionReasons: string[] = []

  const createUsageSelection = (
    usageSource: UsageSource,
    usagePath: UsagePath,
    eventType: string | null,
    rawUsage: unknown,
  ): UsageSelection | null => {
    const normalized = normalizeResponsesUsage(rawUsage)
    if (normalized.rejectionReason !== null) {
      usageRejectionReasons.push(`${eventType ?? 'unknown'}:${usagePath}:${normalized.rejectionReason}`)
      return null
    }

    return {
      usageSource,
      usagePath,
      eventType,
      rawUsage,
      inputTokens: normalized.inputTokens,
      outputTokens: normalized.outputTokens,
      cacheReadTokens: normalized.cacheReadTokens,
      cacheWriteTokens: normalized.cacheWriteTokens,
    }
  }

  const selectUsage = (): UsageSelection => {
    if (completedResponseUsage) return completedResponseUsage
    if (completedTopLevelUsage) return completedTopLevelUsage
    if (latestOtherUsage) return latestOtherUsage
    if (fallbackRetrievedUsage) return fallbackRetrievedUsage

    const rawResponseUsage = (rawResponseBody as { usage?: unknown } | null)?.usage
    if (rawResponseUsage !== undefined) {
      const fromRaw = createUsageSelection('raw-response-body.usage', 'rawResponseBody.usage', terminalEventType, rawResponseUsage)
      if (fromRaw) return fromRaw
    }

    return {
      usageSource: 'none',
      usagePath: 'none',
      eventType: null,
      rawUsage: null,
      inputTokens: null,
      outputTokens: null,
      cacheReadTokens: null,
      cacheWriteTokens: null,
    }
  }

  const isTerminalEvent = (eventType: string): boolean =>
    eventType === 'response.completed' || eventType === 'response.incomplete' || eventType === 'response.failed'

  const shouldTryFallbackRetrieve = (): boolean => {
    if (!responseId) return false
    if (selectUsage().usageSource !== 'none') return false
    if (fallbackRetrieveUsed) return false
    const hasTerminal = Array.from(eventTypeCounts.keys()).some((eventType) => isTerminalEvent(eventType))
    return sawAbort || !hasTerminal
  }

  const tryFallbackRetrieve = async (): Promise<void> => {
    if (!shouldTryFallbackRetrieve() || !responseId) return
    fallbackRetrieveUsed = true
    try {
      const retrieveUrl = `${connection.endpoint.replace(/\/$/, '')}/${encodeURIComponent(responseId)}`
      const retrieveResponse = await fetch(retrieveUrl, {
        method: 'GET',
        headers: connection.headers,
      })

      if (!retrieveResponse.ok) {
        const text = await retrieveResponse.text()
        usageRejectionReasons.push(`fallback-retrieve:http-${retrieveResponse.status}:${text.slice(0, 200)}`)
        return
      }

      const retrieved = await retrieveResponse.json() as { usage?: unknown; response?: { usage?: unknown } } | null
      fallbackRetrieveSucceeded = true
      const fromResponseUsage = createUsageSelection(
        'fallback-retrieve.response.usage',
        'fallbackRetrieve.response.usage',
        'fallback-retrieve',
        retrieved?.response?.usage,
      )
      if (fromResponseUsage) {
        fallbackRetrievedUsage = fromResponseUsage
        fallbackRetrieveUsageFound = true
        fallbackRetrieveUsagePath = 'fallbackRetrieve.response.usage'
        return
      }

      const fromTopLevelUsage = createUsageSelection(
        'fallback-retrieve.usage',
        'fallbackRetrieve.usage',
        'fallback-retrieve',
        retrieved?.usage,
      )
      if (fromTopLevelUsage) {
        fallbackRetrievedUsage = fromTopLevelUsage
        fallbackRetrieveUsageFound = true
        fallbackRetrieveUsagePath = 'fallbackRetrieve.usage'
      }
    } catch (error) {
      usageRejectionReasons.push(`fallback-retrieve:error:${error instanceof Error ? error.message : String(error)}`)
    }
  }

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

    const processSseLine = (rawLine: string): { deltas: string[]; stop: boolean } => {
      const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine
      if (line.length > 0) {
        rawStreamTail.push(line)
        if (rawStreamTail.length > rawStreamTailLimit) rawStreamTail.shift()
      }
      if (!line.startsWith('data: ')) return { deltas: [], stop: false }

      const data = line.slice(6)
      if (data === '[DONE]') {
        sawDoneSentinel = true
        streamEndReason = 'done-sentinel'
        return { deltas: [], stop: true }
      }

      try {
        const event = JSON.parse(data)
        sseEvents.push(event)
        parsedEventCount += 1
        if (typeof event?.type === 'string') {
          eventTypeCounts.set(event.type, (eventTypeCounts.get(event.type) ?? 0) + 1)
        }
        const candidateResponseId = event?.response?.id ?? event?.response_id ?? event?.id ?? null
        if (typeof candidateResponseId === 'string' && candidateResponseId.length > 0) {
          responseIdSeen = true
          responseId = candidateResponseId
        }
        if (event.type === 'response.output_text.delta') {
          return { deltas: [normalizeQuotesInString(event.delta ?? '')], stop: false }
        }

        if (typeof event?.type === 'string' && event.type.startsWith('response.')) {
          terminalEventType = event.type
          terminalEventPayload = event

          if (event.type === 'response.completed') {
            rawResponseBody = event.response ?? null
            if (event?.response?.usage !== undefined) {
              usageBearingEventCount += 1
              const candidateFromResponse = createUsageSelection(
                'response.completed.response.usage',
                'response.completed.response.usage',
                event.type,
                event.response.usage,
              )
              if (candidateFromResponse) completedResponseUsage = candidateFromResponse
            }

            if (event?.usage !== undefined) {
              usageBearingEventCount += 1
              const candidateFromTopLevel = createUsageSelection(
                'response.completed.usage',
                'response.completed.usage',
                event.type,
                event.usage,
              )
              if (candidateFromTopLevel) completedTopLevelUsage = candidateFromTopLevel
            }
          } else {
            const nonTerminalRawUsage = event?.response?.usage ?? event?.usage
            if (nonTerminalRawUsage !== undefined) {
              usageBearingEventCount += 1
              const nonTerminalCandidate = createUsageSelection(
                'response.other',
                event?.response?.usage !== undefined ? 'event.response.usage' : 'event.usage',
                event.type,
                nonTerminalRawUsage,
              )
              if (nonTerminalCandidate) latestOtherUsage = nonTerminalCandidate
            }
          }
        }
      } catch {}

      return { deltas: [], stop: false }
    }

    const processBuffer = (input: string, flushTrailingLine: boolean): { remaining: string; deltas: string[]; stop: boolean } => {
      const lines = input.split('\n')
      let remaining = lines.pop() ?? ''
      const deltas: string[] = []

      for (const line of lines) {
        const result = processSseLine(line)
        deltas.push(...result.deltas)
        if (result.stop) return { remaining, deltas, stop: true }
      }

      if (flushTrailingLine && remaining.length > 0) {
        const result = processSseLine(remaining)
        deltas.push(...result.deltas)
        remaining = ''
        if (result.stop) return { remaining, deltas, stop: true }
      }

      return { remaining, deltas, stop: false }
    }

    try {
      while (true) {
        const { done, value } = await reader.read()

        if (done) {
          sawEof = true
          if (streamEndReason === 'unknown') streamEndReason = 'eof'
          buffer += decoder.decode()
          const finalized = processBuffer(buffer, true)
          for (const delta of finalized.deltas) {
            yield delta
          }
          if (finalized.stop) {
            await tryFallbackRetrieve()
            return
          }
          break
        }

        const decodedChunk = decoder.decode(value, { stream: true })
        if (decodedChunk.length > 0) {
          rawStreamTail.push(decodedChunk)
          if (rawStreamTail.length > rawStreamTailLimit) rawStreamTail.shift()
        }
        buffer += decodedChunk
        const parsed = processBuffer(buffer, false)
        buffer = parsed.remaining
        for (const delta of parsed.deltas) {
          yield delta
        }
        if (parsed.stop) {
          await tryFallbackRetrieve()
          return
        }
      }
    } catch (error) {
      if ((error as { name?: unknown })?.name === 'AbortError') {
        sawAbort = true
        streamEndReason = 'aborted'
        await tryFallbackRetrieve()
      } else {
        streamError = error instanceof Error ? `${error.name}: ${error.message}` : String(error)
        streamEndReason = 'error'
      }
      throw error
    }

    await tryFallbackRetrieve()
  }

  return {
    stream: Stream.fromAsyncIterable(generate(), (e) => classifyUnknownError(e)),
    getUsage(): CallUsage {
      const auth = req.connection._tag === 'Responses' ? req.connection.auth : null
      const selected = selectUsage()
      return buildUsage(
        req.model,
        auth?.type ?? null,
        selected.inputTokens,
        selected.outputTokens,
        selected.cacheReadTokens,
        selected.cacheWriteTokens,
      )
    },
    getCollectorData() {
      const codexVariant = getCodexVariant(req.model, connection.auth)
      const selected = selectUsage()
      const diagnostics = {
        codexVariant,
        providerId: req.model.providerId,
        modelId: req.model.id,
        authType: connection.auth?.type ?? null,
        driverId: 'openai-responses' as const,
        endpoint: connection.endpoint ?? null,
        terminalEventType,
        terminalEventPayload,
        usageSource: selected.usageSource,
        usagePath: selected.usagePath,
        rawUsage: selected.rawUsage,
        parsedInputTokens: selected.inputTokens,
        parsedOutputTokens: selected.outputTokens,
        parsedCacheReadTokens: selected.cacheReadTokens,
        parsedCacheWriteTokens: selected.cacheWriteTokens,
        selectedUsageEventType: selected.eventType,
        usageRejectionReasons,
        usageAbsent: selected.inputTokens === null && selected.outputTokens === null,
        streamEndReason,
        sawDoneSentinel,
        sawEof,
        sawAbort,
        streamError,
        parsedEventCount,
        usageBearingEventCount,
        eventTypeCounts: Object.fromEntries(eventTypeCounts.entries()),
        terminalCompletedCount: eventTypeCounts.get('response.completed') ?? 0,
        terminalIncompleteCount: eventTypeCounts.get('response.incomplete') ?? 0,
        terminalFailedCount: eventTypeCounts.get('response.failed') ?? 0,
        responseIdSeen,
        responseId,
        fallbackRetrieveUsed,
        fallbackRetrieveSucceeded,
        fallbackRetrieveUsageFound,
        fallbackRetrieveUsagePath,
        rawStreamTail,
      }

      logger.info(
        {
          context: 'RESPONSES_FINALIZATION_DEBUG',
          marker: 'RESPONSES_FINALIZATION_DEBUG',
          providerId: req.model.providerId,
          modelId: req.model.id,
          authType: connection.auth?.type ?? null,
          codexVariant,
          parsedEventCount,
          terminalCompletedCount: eventTypeCounts.get('response.completed') ?? 0,
          terminalIncompleteCount: eventTypeCounts.get('response.incomplete') ?? 0,
          terminalFailedCount: eventTypeCounts.get('response.failed') ?? 0,
          usageBearingEventCount,
          usageSource: selected.usageSource,
          streamEndReason,
          sawDoneSentinel,
          sawEof,
          sawAbort,
          responseId,
          fallbackRetrieveUsed,
          fallbackRetrieveSucceeded,
          fallbackRetrieveUsageFound,
          fallbackRetrieveUsagePath,
        },
        '[ResponsesDriver] RESPONSES_FINALIZATION_DEBUG',
      )

      if (usageDiagnosticsEnabled) {
        logger.info(
          { context: 'ResponsesDriverUsageDiagnostics', ...diagnostics },
          '[ResponsesDriver] usage diagnostics',
        )
      }

      return CollectorData.Responses({ rawRequestBody, rawResponseBody, sseEvents, diagnostics })
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