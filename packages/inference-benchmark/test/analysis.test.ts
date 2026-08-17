import { describe, expect, it } from "vitest"
import { analyzeTrial } from "../src/analysis"
import type { RequestObservation, TrialObservation } from "../src/domain"

const invalidWithEvidence: RequestObservation = {
  requestId: "request-1",
  outcome: "invalid",
  status: 200,
  headersMs: 1,
  ttftMs: 10,
  completedMs: 20,
  outputText: "wrong answer",
  toolCalls: [],
  finishReason: "tool_calls",
  terminal: {
    usage: { promptTokens: 100, completionTokens: 10, totalTokens: 110, cachedPromptTokens: 0 },
    timings: { cacheTokens: 0, evaluatedPromptTokens: 100, promptMs: 50, generatedTokens: 10, generationMs: 20 },
  },
  events: [],
  error: "wrong tool call",
}

function observation(pattern: TrialObservation["trial"]["pattern"]): TrialObservation {
  return {
    trial: {
      id: "trial-1",
      pattern,
      criteria: ["responsiveness", "prefill", "decode", "memory-usage", "distribution"],
      checkpoint: "checkpoint",
      repetition: 0,
      state: "cache-disjoint",
      requests: [],
    },
    startedAt: new Date(0).toISOString(),
    makespanMs: 20,
    requests: [invalidWithEvidence],
  }
}

describe("trial analysis", () => {
  it("measures protocol-valid context scaling evidence separately from semantic validity", () => {
    const analysis = analyzeTrial(observation("context-scaling"))
    expect(analysis.measuredRequests).toBe(1)
    expect(analysis.validRequests).toBe(0)
    expect(analysis.outcomes.invalid).toBe(1)
    expect(analysis.promptTokens?.median).toBe(100)
    expect(analysis.prefillTokensPerSecond?.median).toBe(2_000)
  })

  it("continues to exclude semantically invalid evidence from standard workloads", () => {
    const analysis = analyzeTrial(observation("single-request"))
    expect(analysis.measuredRequests).toBe(0)
    expect(analysis.prefillTokensPerSecond).toBeUndefined()
  })
})
