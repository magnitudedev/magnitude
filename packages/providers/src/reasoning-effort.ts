/**
 * Automatic low-effort reasoning configuration.
 *
 * Magnitude provides its own think tool, so we always set the
 * lowest available reasoning effort on models that support it.
 * This reduces latency and cost from redundant model thinking.
 */

interface EffortOptions {
  /** Options to deep-merge into the BAML client options */
  optionsMerge: Record<string, any>
  /** Human-readable label for logging */
  label: string
}

/**
 * Determine the lowest effort options for a given provider + model.
 * Returns null if the model doesn't support configurable reasoning effort.
 */
export function getLowestEffortOptions(
  providerId: string,
  modelId: string,
  bamlProvider: string,
): EffortOptions | null {
  const id = modelId.toLowerCase()

  // --- GitHub Copilot (must check before generic openai/anthropic) ---
  if (providerId === 'github-copilot') {
    // Claude models on Copilot use thinking_budget as a top-level param
    if (id.includes('claude')) {
      return {
        optionsMerge: { thinking_budget: 1 },
        label: 'Copilot Claude thinking_budget=1',
      }
    }
    // o-series on Copilot — lowest is "low"
    if (id.includes('o3') || id.includes('o4')) {
      return {
        optionsMerge: { reasoning_effort: 'low' },
        label: 'Copilot reasoning_effort=low',
      }
    }
    // Codex models on Copilot — lowest supported is "low"
    if (id.includes('codex')) {
      return {
        optionsMerge: { reasoning_effort: 'low' },
        label: 'Copilot reasoning_effort=low',
      }
    }
    // GPT-5 base on Copilot — lowest supported is "low"
    if (id.includes('gpt-5') && !id.includes('5.')) {
      return {
        optionsMerge: { reasoning_effort: 'low' },
        label: 'Copilot reasoning_effort=low',
      }
    }
    // GPT-5.1+, GPT-5.2+ non-Codex already default to "none" — skip
    // Gemini on Copilot: no reasoning params supported
    return null
  }

  // --- Z.AI (disable thinking entirely — we provide our own reasoning) ---
  if (providerId === 'zai' || providerId === 'zai-coding-plan') {
    return {
      optionsMerge: { thinking: { type: 'disabled' } },
      label: 'Z.AI thinking=disabled',
    }
  }

  // --- Moonshot / Kimi (disable thinking — we provide our own reasoning) ---
  if (providerId === 'moonshotai' || providerId === 'kimi-for-coding') {
    return {
      optionsMerge: { thinking: { type: 'disabled' } },
      label: 'Kimi thinking=disabled',
    }
  }

  // --- Anthropic Claude (direct API, Vertex Anthropic) ---
  // Opus 4.6, Sonnet 4.6, Opus 4.5 support effort (Sonnet 4.5 does NOT)
  if (
    (bamlProvider === 'anthropic' && !providerId.startsWith('minimax')) &&
    isClaudeWithEffort(id)
  ) {
    return {
      optionsMerge: { output_config: { effort: 'low' } },
      label: 'Anthropic effort=low',
    }
  }

  // --- Vertex AI Anthropic ---
  if (providerId === 'google-vertex-anthropic' && isClaudeWithEffort(id)) {
    return {
      optionsMerge: { output_config: { effort: 'low' } },
      label: 'Vertex Anthropic effort=low',
    }
  }

  // --- Amazon Bedrock (Anthropic models) ---
  if (bamlProvider === 'aws-bedrock' && isClaudeWithEffort(id)) {
    return {
      optionsMerge: {
        additional_model_request_fields: {
          output_config: { effort: 'low' },
        },
      },
      label: 'Bedrock Anthropic effort=low',
    }
  }

  // --- OpenAI (direct API, Responses API) ---
  // o-series (o3, o4-mini) are dedicated reasoning models — lowest is "low"
  // Codex models lowest supported is "low" (none is NOT valid for Codex)
  // GPT-5 base defaults to "medium" — lowest supported is "low"
  // GPT-5.1+, GPT-5.2+ non-Codex already default to "none" — skip
  if (bamlProvider === 'openai' || bamlProvider === 'openai-responses') {
    if (id.includes('o3') || id.includes('o4')) {
      return {
        optionsMerge: { reasoning: { effort: 'low' } },
        label: 'OpenAI reasoning.effort=low',
      }
    }
    if (id.includes('codex')) {
      return {
        optionsMerge: { reasoning: { effort: 'low' } },
        label: 'OpenAI reasoning.effort=low',
      }
    }
    // GPT-5 base (not 5.1, 5.2, etc.) defaults to "medium"
    if (id.includes('gpt-5') && !id.includes('5.')) {
      return {
        optionsMerge: { reasoning: { effort: 'low' } },
        label: 'OpenAI reasoning.effort=low',
      }
    }
    // GPT-5.1, GPT-5.2, etc. already default to "none" — don't touch
  }

  // --- Google Gemini 3 (thinkingLevel) ---
  if (
    (bamlProvider === 'vertex-ai' || bamlProvider === 'google-ai') &&
    id.includes('gemini-3')
  ) {
    // Flash supports MINIMAL, Pro only supports LOW as lowest
    const level = id.includes('flash') ? 'MINIMAL' : 'LOW'
    return {
      optionsMerge: {
        generationConfig: {
          thinkingConfig: { thinkingLevel: level },
        },
      },
      label: `Gemini 3 thinkingLevel=${level}`,
    }
  }

  // --- Google Gemini 2.5 (thinkingBudget) ---
  if (
    (bamlProvider === 'vertex-ai' || bamlProvider === 'google-ai') &&
    id.includes('gemini-2.5')
  ) {
    return {
      optionsMerge: {
        generationConfig: {
          thinkingConfig: { thinkingBudget: 0 },
        },
      },
      label: 'Gemini 2.5 thinkingBudget=0',
    }
  }

  return null
}

/**
 * Get the reasoning effort value for manual Codex/Copilot HTTP requests.
 * Used by codex-stream.ts which bypasses BAML's request building.
 * These streams are only used for Codex models which default to "medium".
 * Returns null if no effort should be set.
 */
export function getCodexReasoningEffort(modelId: string): string | null {
  const id = modelId.toLowerCase()
  if (id.includes('codex') || id.includes('gpt-5') || id.includes('o3') || id.includes('o4')) {
    return 'low'
  }
  return null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Check if a model ID matches a Claude model that supports the effort parameter.
 * Per Anthropic docs: effort is supported by Opus 4.6, Sonnet 4.6, and Opus 4.5.
 * Sonnet 4.5 does NOT support effort. */
function isClaudeWithEffort(id: string): boolean {
  // Opus 4.6
  if (id.includes('opus-4-6') || id.includes('opus-4.6')) return true
  // Sonnet 4.6
  if (id.includes('sonnet-4-6') || id.includes('sonnet-4.6')) return true
  // Opus 4.5 — but NOT Sonnet 4.5
  if (
    (id.includes('opus-4-5') || id.includes('opus-4.5')) &&
    !id.includes('sonnet')
  ) return true
  return false
}

