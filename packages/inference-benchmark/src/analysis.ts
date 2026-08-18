import type { MetricSummary, Outcome, TrialAnalysis, TrialObservation } from "./domain"

function quantile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return Number.NaN
  const position = (sorted.length - 1) * q
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return sorted[lower]!
  const weight = position - lower
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight
}

export function summarize(values: readonly number[]): MetricSummary | undefined {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b)
  if (finite.length === 0) return undefined
  const median = quantile(finite, 0.5)
  const deviations = finite.map((value) => Math.abs(value - median)).sort((a, b) => a - b)
  return {
    count: finite.length,
    min: finite[0]!,
    max: finite.at(-1)!,
    mean: finite.reduce((sum, value) => sum + value, 0) / finite.length,
    median,
    p90: quantile(finite, 0.9),
    p95: quantile(finite, 0.95),
    p99: quantile(finite, 0.99),
    medianAbsoluteDeviation: quantile(deviations, 0.5),
  }
}

export function analyzeTrial(observation: TrialObservation): TrialAnalysis {
  const outcomes: Record<Outcome, number> = {
    valid: 0,
    invalid: 0,
    rejected: 0,
    "protocol-error": 0,
    error: 0,
    timeout: 0,
    cancelled: 0,
  }
  for (const request of observation.requests) outcomes[request.outcome]++
  const valid = observation.requests.filter((request) => request.outcome === "valid")
  // Context scaling measures engine performance at a selected depth. A model can
  // produce a semantically invalid BFCL answer while still returning complete,
  // internally consistent terminal timing evidence. Preserve that evidence while
  // continuing to report semantic validity separately.
  const measured = observation.trial.pattern === "context-scaling"
    ? observation.requests.filter((request) =>
        (request.outcome === "valid" || request.outcome === "invalid") && request.terminal !== undefined)
    : valid
  const ttft = measured.flatMap((request) => request.ttftMs === undefined ? [] : [request.ttftMs])
  const prefill = measured.flatMap((request) => {
    const timings = request.terminal?.timings
    return timings && timings.evaluatedPromptTokens > 0 && timings.promptMs > 0
      ? [1_000 * timings.evaluatedPromptTokens / timings.promptMs]
      : []
  })
  const decode = measured.flatMap((request) => {
    const timings = request.terminal?.timings
    return timings && timings.generatedTokens > 0 && timings.generationMs > 0
      ? [1_000 * timings.generatedTokens / timings.generationMs]
      : []
  })
  const cacheReuse = measured.flatMap((request) => {
    const usage = request.terminal?.usage
    return usage && usage.promptTokens > 0 ? [usage.cachedPromptTokens / usage.promptTokens] : []
  })
  const promptTokenCounts = measured.flatMap((request) => request.terminal ? [request.terminal.usage.promptTokens] : [])
  const completionTokenCounts = measured.flatMap((request) => request.terminal ? [request.terminal.usage.completionTokens] : [])
  const completion = measured.map((request) => request.completedMs)
  const measuredSeconds = observation.makespanMs / 1_000
  const completionTokens = measured.reduce((sum, request) => sum + (request.terminal?.usage.completionTokens ?? 0), 0)
  return {
    trialId: observation.trial.id,
    pattern: observation.trial.pattern,
    measuredRequests: measured.length,
    validRequests: valid.length,
    outcomes,
    responsivenessMs: summarize(ttft),
    prefillTokensPerSecond: summarize(prefill),
    decodeTokensPerSecond: summarize(decode),
    completionMs: summarize(completion),
    achievedRequestsPerSecond: measuredSeconds > 0 ? measured.length / measuredSeconds : undefined,
    achievedCompletionTokensPerSecond: measuredSeconds > 0 ? completionTokens / measuredSeconds : undefined,
    cacheReuseRatio: summarize(cacheReuse),
    promptTokens: summarize(promptTokenCounts),
    completionTokens: summarize(completionTokenCounts),
    memory: observation.memory,
  }
}
