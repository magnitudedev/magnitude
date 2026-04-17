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

  // --- Magnitude (provider-specific reasoning policy) ---
  if (providerId === 'magnitude') {
    if (id.includes('minimax')) {
      return {
        optionsMerge: { reasoning_effort: 'low' },
        label: 'Magnitude reasoning_effort=low',
      }
    }
    return {
      optionsMerge: { reasoning_effort: 'none' },
      label: 'Magnitude reasoning_effort=none',
    }
  }

  // --- Fireworks AI (disable reasoning where supported) ---
  // GLM models support reasoning_effort: "none"
  // Kimi models on Fireworks also support disabling via reasoning_effort: "none"
  // MiniMax requires reasoning enabled (minimum "low") — cannot disable
  if (providerId === 'fireworks-ai') {
    if (id.includes('minimax')) {
      return {
        optionsMerge: { reasoning_effort: 'low' },
        label: 'Fireworks reasoning_effort=low',
      }
    }
    // GLM, Kimi, and other models: disable reasoning entirely
    return {
      optionsMerge: { reasoning_effort: 'none' },
      label: 'Fireworks reasoning_effort=none',
    }
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

  // --- Anthropic Claude (direct API) ---
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

export function canUseGrammarWithStreaming(
  providerId: string,
  modelId: string,
  bamlProvider: string,
): boolean {
  const id = modelId.toLowerCase()

  // Fireworks: MiniMax and GPT-OSS cannot disable reasoning (min "low")
  // Grammar + streaming is broken for these models
  if (providerId === 'fireworks-ai') {
    if (id.includes('minimax') || id.includes('gpt-oss')) {
      return false
    }
  }

  return true
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

