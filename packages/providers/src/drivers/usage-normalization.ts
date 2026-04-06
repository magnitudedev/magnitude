type UsageRejectionReason =
  | 'missing-usage-object'
  | 'missing-numeric-token-fields'

export interface NormalizedUsage {
  readonly inputTokens: number | null
  readonly outputTokens: number | null
  readonly cacheReadTokens: number | null
  readonly cacheWriteTokens: number | null
  readonly rejectionReason: UsageRejectionReason | null
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : null
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' ? value : null
}

function pickNumber(record: Record<string, unknown>, keys: readonly string[]): number | null {
  for (const key of keys) {
    const value = asNumber(record[key])
    if (value !== null) return value
  }
  return null
}

export function normalizeResponsesUsage(rawUsage: unknown): NormalizedUsage {
  const usage = asRecord(rawUsage)
  if (!usage) {
    return {
      inputTokens: null,
      outputTokens: null,
      cacheReadTokens: null,
      cacheWriteTokens: null,
      rejectionReason: 'missing-usage-object',
    }
  }

  const inputTokens = pickNumber(usage, ['input_tokens', 'inputTokens'])
  const outputTokens = pickNumber(usage, ['output_tokens', 'outputTokens'])
  const inputTokensDetails = asRecord(usage.input_tokens_details) ?? asRecord(usage.inputTokensDetails)
  const cachedTokens = pickNumber(usage, ['cachedInputTokens'])
    ?? (inputTokensDetails ? pickNumber(inputTokensDetails, ['cached_tokens', 'cachedTokens']) : null)

  if (inputTokens === null && outputTokens === null) {
    return {
      inputTokens: null,
      outputTokens: null,
      cacheReadTokens: null,
      cacheWriteTokens: null,
      rejectionReason: 'missing-numeric-token-fields',
    }
  }

  return {
    inputTokens: inputTokens === null ? null : inputTokens + (cachedTokens ?? 0),
    outputTokens,
    cacheReadTokens: cachedTokens,
    cacheWriteTokens: null,
    rejectionReason: null,
  }
}

export function normalizeAnthropicUsage(rawUsage: unknown): NormalizedUsage {
  const usage = asRecord(rawUsage)
  if (!usage) {
    return {
      inputTokens: null,
      outputTokens: null,
      cacheReadTokens: null,
      cacheWriteTokens: null,
      rejectionReason: 'missing-usage-object',
    }
  }

  const inputTokens = asNumber(usage.input_tokens)
  const outputTokens = asNumber(usage.output_tokens)
  const cacheReadTokens = asNumber(usage.cache_read_input_tokens)
  const cacheWriteTokens = asNumber(usage.cache_creation_input_tokens)

  if (inputTokens === null && outputTokens === null) {
    return {
      inputTokens: null,
      outputTokens: null,
      cacheReadTokens: null,
      cacheWriteTokens: null,
      rejectionReason: 'missing-numeric-token-fields',
    }
  }

  return {
    inputTokens: inputTokens === null ? null : inputTokens + (cacheReadTokens ?? 0) + (cacheWriteTokens ?? 0),
    outputTokens,
    cacheReadTokens,
    cacheWriteTokens,
    rejectionReason: null,
  }
}