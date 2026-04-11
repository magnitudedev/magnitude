function validateTokenCount(tokens: number): number | null {
  if (tokens <= 0) return null
  return tokens
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' ? value : null
}

export type ParsedRawUsage = {
  inputTokens: number | null
  outputTokens: number | null
  cacheReadTokens: number | null
  cacheWriteTokens: number | null
}

type UsageCarrier = {
  inputTokens?: number
  cachedInputTokens?: number
  outputTokens?: number
}

type CollectorLike = {
  last?: {
    calls: ReadonlyArray<{
      httpResponse?: { body?: { json?(): unknown } }
      sseResponses?(): ReadonlyArray<{ json?(): unknown }>
    }>
  }
  usage?: UsageCarrier
}

/**
 * Provider-agnostic usage parser for raw response/event usage objects.
 * Supports Anthropic-style, OpenAI-compatible, and Ollama fields.
 */
export function parseRawUsage(rawUsage: unknown): ParsedRawUsage {
  if (!rawUsage || typeof rawUsage !== 'object' || Array.isArray(rawUsage)) {
    return { inputTokens: null, outputTokens: null, cacheReadTokens: null, cacheWriteTokens: null }
  }

  const usage = rawUsage as Record<string, unknown>

  // Anthropic-style
  const anthropicInput = numberOrNull(usage.input_tokens)
  const anthropicOutput = numberOrNull(usage.output_tokens)
  const cacheCreationInput = numberOrNull(usage.cache_creation_input_tokens)
  const cacheReadInput = numberOrNull(usage.cache_read_input_tokens)

  // OpenAI-compatible style (LM Studio / llama.cpp compatible endpoints)
  const openAIPromptTokens = numberOrNull(usage.prompt_tokens)
  const openAICompletionTokens = numberOrNull(usage.completion_tokens)

  // Ollama native-style counters
  const ollamaPromptEvalCount = numberOrNull(usage.prompt_eval_count)
  const ollamaEvalCount = numberOrNull(usage.eval_count)

  let inputTokens: number | null = null
  let outputTokens: number | null = null
  let cacheReadTokens: number | null = null
  let cacheWriteTokens: number | null = null

  if (anthropicInput !== null) {
    const total = anthropicInput + (cacheCreationInput ?? 0) + (cacheReadInput ?? 0)
    inputTokens = validateTokenCount(total)
    cacheReadTokens = cacheReadInput
    cacheWriteTokens = cacheCreationInput
  } else if (openAIPromptTokens !== null) {
    inputTokens = validateTokenCount(openAIPromptTokens)
  } else if (ollamaPromptEvalCount !== null) {
    inputTokens = validateTokenCount(ollamaPromptEvalCount)
  }

  if (anthropicOutput !== null) {
    outputTokens = anthropicOutput
  } else if (openAICompletionTokens !== null) {
    outputTokens = openAICompletionTokens
  } else if (ollamaEvalCount !== null) {
    outputTokens = ollamaEvalCount
  }

  return { inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens }
}

export function extractUsageFromCollectorData(collector: CollectorLike): ParsedRawUsage {
  let inputTokens: number | null = null
  let outputTokens: number | null = null
  let cacheReadTokens: number | null = null
  let cacheWriteTokens: number | null = null

  const lastCall = collector.last?.calls.at(-1)

  if (lastCall) {
    // Strategy 1: Extract from HTTP response body JSON usage
    try {
      const rawBody = lastCall.httpResponse?.body?.json?.()
      const rawUsage =
        rawBody && typeof rawBody === 'object' && 'usage' in (rawBody as Record<string, unknown>)
          ? (rawBody as Record<string, unknown>).usage
          : null
      const parsed = parseRawUsage(rawUsage)
      if (parsed.inputTokens !== null) inputTokens = parsed.inputTokens
      if (parsed.outputTokens !== null) outputTokens = parsed.outputTokens
      if (parsed.cacheReadTokens !== null) cacheReadTokens = parsed.cacheReadTokens
      if (parsed.cacheWriteTokens !== null) cacheWriteTokens = parsed.cacheWriteTokens
    } catch {}

    // Strategy 2: Extract from SSE events usage payloads
    if (inputTokens === null) {
      try {
        const sseResponses = lastCall.sseResponses?.() ?? null
        if (Array.isArray(sseResponses)) {
          for (const sse of sseResponses) {
            const data = sse.json?.() as {
              message?: { usage?: unknown }
              response?: { usage?: unknown }
              usage?: unknown
            } | null
            if (!data) continue

            const candidates = [data.usage, data.message?.usage, data.response?.usage]
            for (const candidate of candidates) {
              const parsed = parseRawUsage(candidate)
              if (inputTokens === null && parsed.inputTokens !== null) inputTokens = parsed.inputTokens
              if (outputTokens === null && parsed.outputTokens !== null) outputTokens = parsed.outputTokens
              if (cacheReadTokens === null && parsed.cacheReadTokens !== null) cacheReadTokens = parsed.cacheReadTokens
              if (cacheWriteTokens === null && parsed.cacheWriteTokens !== null) cacheWriteTokens = parsed.cacheWriteTokens
            }
          }
        }
      } catch {}
    }
  }

  // Strategy 3: Fallback to collector-level usage
  if (inputTokens === null) {
    const usage = collector.usage
    if (usage) {
      const total = (usage.inputTokens ?? 0) + (usage.cachedInputTokens ?? 0)
      inputTokens = validateTokenCount(total)
    }
  }
  if (outputTokens === null) {
    const usage = collector.usage
    if (usage && typeof usage.outputTokens === 'number') outputTokens = usage.outputTokens
  }

  return { inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens }
}
